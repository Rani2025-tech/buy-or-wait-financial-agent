"""
decision_engine.py -- Deterministic affordability decision engine.

Evaluates one request at a time and returns a structured result.
Uses data_loader.Dataset and financial_engine functions exclusively.
Does NOT call any LLM, OCR, or external service.

Evaluation order (per problem spec):
  1. Full payment now (no spending changes)
  2. Full payment with spending changes
  3. Installments (from supplied options only)
  4. Partial payment (exactly 2 payments)
  5. Wait / affordable later
  6. Not affordable

Plan ranking (per problem spec):
  1. Completes by desired_completion_date
  2. No spending changes needed
  3. Minimize total amount paid
  4. Start payment earlier
  5. Fewer payments
  6. Lowest payment_option_id (tie-breaker)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from data_loader import Dataset, FinancialProfile, Request
from financial_engine import (
    FlexibleExpense,
    _round2,
    build_projection,
    compute_amount_safe_to_pay,
    earliest_full_payment_date,
    get_flexible_expenses,
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str       # affordable_now | affordable_with_plan | affordable_later | not_affordable
    recommended_payment_method: str # full_payment | partial_payment | installments | wait | not_recommended
    payment_plan: str               # "YYYY-MM-DD:amount|..." or "none"
    earliest_date_for_full_payment: str  # "YYYY-MM-DD" or ""
    spending_changes_needed: str    # "stop:eid|reduce_to:eid:amt" or "none"
    decision_explanation: str = ""  # Grounded explanation of recommendation
    error: Optional[str] = None     # set if image amounts are missing


# ---------------------------------------------------------------------------
# Payment plan formatting
# ---------------------------------------------------------------------------

def _format_plan(payments: List[Tuple[date, float]]) -> str:
    """Format a list of (date, amount) into the official pipe-separated string."""
    return "|".join(f"{d}:{_round2(a)}" for d, a in sorted(payments))


def _format_spending_changes(changes: List[str]) -> str:
    return "|".join(changes) if changes else "none"


def _format_num(val: float) -> str:
    """Format numeric amounts with comma separation."""
    if val == int(val):
        return f"{int(val):,}"
    else:
        return f"{val:,.2f}"


def _format_date(d: date) -> str:
    """Format date as e.g. 15 June 2024."""
    return f"{d.day} {d.strftime('%B')} {d.year}"


def generate_explanation(
    req: Request,
    profile: FinancialProfile,
    status: str,
    method: str,
    plan_str: str,
    earliest_str: str,
    changes_str: str,
    safe_amount: float,
    dataset: Dataset,
) -> str:
    """Deterministic explanation generation grounded in calculated fields."""
    cur = profile.home_currency
    min_bal = profile.minimum_balance_to_keep
    req_amt = req.requested_amount

    if status == "affordable_now":
        return f"Pay {cur} {_format_num(req_amt)} today. This leaves at least {cur} {_format_num(min_bal)} available over the next 90 days."

    elif method == "installments" and plan_str != "none":
        parts = plan_str.split("|")
        n_inst = len(parts)
        try:
            first_date_str, first_amt_str = parts[0].split(":")
            first_dt = date.fromisoformat(first_date_str)
            inst_amt = float(first_amt_str)
            return f"Use {n_inst} installments of {cur} {_format_num(inst_amt)}, starting {_format_date(first_dt)}. This leaves at least {cur} {_format_num(min_bal)} available."
        except Exception:
            return f"Use installments to pay {cur} {_format_num(req_amt)}. This leaves at least {cur} {_format_num(min_bal)} available."

    elif method == "partial_payment" and plan_str != "none":
        parts = plan_str.split("|")
        try:
            p1_dt_str, p1_amt_str = parts[0].split(":")
            p2_dt_str, p2_amt_str = parts[1].split(":")
            p1_amt = float(p1_amt_str)
            p2_amt = float(p2_amt_str)
            p2_dt = date.fromisoformat(p2_dt_str)
            return f"Pay {cur} {_format_num(p1_amt)} today and the remaining {cur} {_format_num(p2_amt)} on {_format_date(p2_dt)}. This completes the full request and keeps the {cur} {_format_num(min_bal)} minimum protected."
        except Exception:
            return f"Pay partially today and the remainder later. This keeps the {cur} {_format_num(min_bal)} minimum protected."

    elif method == "wait" or status == "affordable_later":
        if earliest_str:
            try:
                e_dt = date.fromisoformat(earliest_str)
                return f"Pay {cur} {_format_num(req_amt)} in full on {_format_date(e_dt)}. Paying earlier would take the balance below the {cur} {_format_num(min_bal)} minimum."
            except Exception:
                pass
        return f"Wait until funds settle before paying {cur} {_format_num(req_amt)} in full. Paying earlier would put the {cur} {_format_num(min_bal)} minimum at risk."

    elif status == "affordable_with_plan" and method == "full_payment":
        if changes_str and changes_str != "none":
            desc_parts = []
            for c in changes_str.split("|"):
                if c.startswith("stop:"):
                    eid = c[5:]
                    ev = dataset.get_event(eid)
                    desc = ev.description.lower() if ev else eid
                    desc_parts.append(f"stop the {desc}")
                elif c.startswith("reduce_to:"):
                    cp = c.split(":")
                    eid = cp[1]
                    amt = float(cp[2])
                    ev = dataset.get_event(eid)
                    desc = ev.description.lower() if ev else eid
                    desc_parts.append(f"reduce the {desc} to {cur} {_format_num(amt)}")
            changes_text = " and ".join(desc_parts).capitalize()
            return f"{changes_text}, then pay {cur} {_format_num(req_amt)} today. This leaves at least {cur} {_format_num(min_bal)} available."
        return f"Pay {cur} {_format_num(req_amt)} with adjusted spending. This leaves at least {cur} {_format_num(min_bal)} available."

    else:  # not_affordable / not_recommended
        if req.desired_completion_date:
            return f"Do not make this payment by {_format_date(req.desired_completion_date)}. None of the available options keeps the {cur} {_format_num(min_bal)} minimum protected."
        else:
            return f"Do not proceed with the {cur} {_format_num(req_amt)} request. Although {cur} {_format_num(safe_amount)} is available today, the full amount cannot be completed safely within 90 days."


# ---------------------------------------------------------------------------
# Installment schedule helpers
# ---------------------------------------------------------------------------

def _installment_schedule(option) -> List[Tuple[date, float]]:
    """Return the full payment schedule for a PaymentOption."""
    payments = []
    d = option.first_payment_date
    for i in range(option.number_of_payments):
        payments.append((d, option.payment_amount))
        if option.payment_frequency_days:
            d = d + timedelta(days=option.payment_frequency_days)
    return payments


def _installment_safe(
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: date,
    schedule: List[Tuple[date, float]],
    stopped_event_ids: Optional[List[str]] = None,
    reduced_events: Optional[Dict[str, float]] = None,
) -> bool:
    """Return True if the installment schedule keeps the spendable balance safe on payment dates."""
    proj = build_projection(
        dataset, profile, request_date,
        extra_debits=schedule,
        stopped_event_ids=stopped_event_ids,
        reduced_events=reduced_events,
    )
    if not schedule:
        return False
    return all(proj.spendable_on(d) >= profile.minimum_balance_to_keep for d, _ in schedule)


def _installment_within_deadline(schedule: List[Tuple[date, float]], deadline: date) -> bool:
    """Return True if the last payment is on or before the deadline."""
    if not schedule:
        return False
    last_date = max(d for d, _ in schedule)
    return last_date <= deadline


# ---------------------------------------------------------------------------
# Spending-change combination helpers
# ---------------------------------------------------------------------------

def _build_reduced_events_map(
    dataset: Dataset,
    profile: FinancialProfile,
    changes: List[FlexibleExpense],
    request_date: date,
) -> Dict[str, float]:
    """
    Build the reduced_events dict for build_projection from a list of
    FlexibleExpense items that are being reduced.
    Maps event_id -> new_amount_home_currency.
    """
    from financial_engine import convert_amount, fx_ref_date
    result: Dict[str, float] = {}
    for fe in changes:
        if fe.can_reduce and fe.reduce_to is not None and fe.reduce_to_home is not None:
            result[fe.event.event_id] = fe.reduce_to_home
    return result


def _spending_change_strings(
    stops: List[FlexibleExpense],
    reduces: List[FlexibleExpense],
) -> List[str]:
    """Format spending changes into the official string format."""
    parts = []
    for fe in stops:
        parts.append(f"stop:{fe.event.event_id}")
    for fe in reduces:
        if fe.reduce_to is not None:
            parts.append(f"reduce_to:{fe.event.event_id}:{_round2(fe.reduce_to)}")
    return parts


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_request(request_id: str, dataset: Dataset) -> DecisionResult:
    """
    Evaluate a single request and return a DecisionResult.

    Follows the official evaluation order and ranking rules.
    Returns an error result if required image amounts are missing.
    """
    req = dataset.get_request(request_id)
    if req is None:
        return DecisionResult(
            request_id=request_id,
            amount_safe_to_pay=0.0,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=f"Request {request_id} not found in records.",
            error=f"Request {request_id} not found",
        )

    profile = dataset.get_profile(req.user_id)
    if profile is None:
        return DecisionResult(
            request_id=request_id,
            amount_safe_to_pay=0.0,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=f"User profile {req.user_id} not found in records.",
            error=f"Profile for {req.user_id} not found",
        )

    try:
        return _evaluate(req, profile, dataset)
    except ValueError as exc:
        # Missing image amount — return partial result with error flag
        safe = _safe_amount_best_effort(req, profile, dataset)
        cur = profile.home_currency
        min_bal = profile.minimum_balance_to_keep
        expl = f"Do not proceed with the {cur} {_format_num(req.requested_amount)} request. Required transaction documentation is pending verification."
        return DecisionResult(
            request_id=request_id,
            amount_safe_to_pay=safe,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment="",
            spending_changes_needed="none",
            decision_explanation=expl,
            error=f"Missing image amount: {exc}",
        )


def _safe_amount_best_effort(req: Request, profile: FinancialProfile, dataset: Dataset) -> float:
    """Compute safe_to_pay ignoring events with missing image amounts."""
    try:
        return compute_amount_safe_to_pay(dataset, profile, req.request_date, req.requested_amount)
    except Exception:
        return 0.0


def _evaluate(req: Request, profile: FinancialProfile, dataset: Dataset) -> DecisionResult:
    """Main evaluation logic. May raise ValueError for missing image amounts."""
    accepted_methods = set(profile.payment_methods_user_will_consider)
    deadline = req.desired_completion_date
    request_date = req.request_date
    requested = req.requested_amount

    # --- Baseline: amount_safe_to_pay (no spending changes) ---
    safe_now = compute_amount_safe_to_pay(dataset, profile, request_date, requested)

    # --- Earliest full payment date (no spending changes) ---
    earliest_no_changes = earliest_full_payment_date(
        dataset, profile, request_date, requested
    )

    # Collect all candidate plans; pick best by ranking rules at the end
    candidates: List[_Plan] = []

    # ---------------------------------------------------------------
    # PATH 1: Full payment now (no spending changes)
    # ---------------------------------------------------------------
    if "full_payment" in accepted_methods and safe_now >= requested:
        candidates.append(_Plan(
            status="affordable_now",
            method="full_payment",
            payments=[(request_date, requested)],
            spending_changes=[],
            total_paid=requested,
            option_id=None,
        ))

    # ---------------------------------------------------------------
    # PATH 2: Full payment with spending changes
    # ---------------------------------------------------------------
    if "full_payment" in accepted_methods and safe_now < requested:
        flex_expenses = get_flexible_expenses(dataset, profile, request_date)
        spending_plan = _find_spending_changes_for_full_payment(
            dataset, profile, req, flex_expenses, request_date, requested
        )
        if spending_plan is not None:
            stops, reduces, safe_with_changes = spending_plan
            if safe_with_changes >= requested:
                candidates.append(_Plan(
                    status="affordable_with_plan",
                    method="full_payment",
                    payments=[(request_date, requested)],
                    spending_changes=_spending_change_strings(stops, reduces),
                    total_paid=requested,
                    option_id=None,
                ))

    # ---------------------------------------------------------------
    # PATH 3: Installments (from supplied options only)
    # ---------------------------------------------------------------
    if "installments" in accepted_methods:
        max_months = profile.max_installment_months
        options = dataset.get_payment_options_for_request(req.request_id)
        for opt in options:
            if opt.payment_method != "installments":
                continue
            # Respect max_installment_months
            if max_months is not None and opt.number_of_payments > max_months:
                continue
            schedule = _installment_schedule(opt)
            if not schedule:
                continue
            if _installment_safe(dataset, profile, request_date, schedule):
                candidates.append(_Plan(
                    status="affordable_with_plan",
                    method="installments",
                    payments=schedule,
                    spending_changes=[],
                    total_paid=opt.total_payable_amount,
                    option_id=opt.payment_option_id,
                ))
            else:
                # Try with flexible spending changes if needed
                flex_expenses = get_flexible_expenses(dataset, profile, request_date)
                spending_plan = _find_spending_changes_for_installments(
                    dataset, profile, req, flex_expenses, request_date, schedule
                )
                if spending_plan is not None:
                    stops, reduces = spending_plan
                    candidates.append(_Plan(
                        status="affordable_with_plan",
                        method="installments",
                        payments=schedule,
                        spending_changes=_spending_change_strings(stops, reduces),
                        total_paid=opt.total_payable_amount,
                        option_id=opt.payment_option_id,
                    ))

    # ---------------------------------------------------------------
    # PATH 4: Partial payment (exactly 2 payments)
    # ---------------------------------------------------------------
    if (
        req.allows_partial_payment
        and "partial_payment" in accepted_methods
        and 0 < safe_now < requested
    ):
        remainder = _round2(requested - safe_now)
        # Second payment occurs on earliest_date_for_full_payment
        if (
            earliest_no_changes is not None
            and earliest_no_changes <= deadline
            and earliest_no_changes > request_date
        ):
            two_pay = [(request_date, safe_now), (earliest_no_changes, remainder)]
            proj = build_projection(dataset, profile, request_date, extra_debits=two_pay)
            if proj.is_safe():
                candidates.append(_Plan(
                    status="affordable_with_plan",
                    method="partial_payment",
                    payments=two_pay,
                    spending_changes=[],
                    total_paid=requested,
                    option_id=None,
                ))

    # ---------------------------------------------------------------
    # PATH 5: Wait / affordable later
    # ---------------------------------------------------------------
    if "full_payment" in accepted_methods and earliest_no_changes is not None:
        if earliest_no_changes > request_date:
            candidates.append(_Plan(
                status="affordable_later",
                method="wait",
                payments=[(earliest_no_changes, requested)],
                spending_changes=[],
                total_paid=requested,
                option_id=None,
            ))

    # ---------------------------------------------------------------
    # Rank candidates and pick best
    # ---------------------------------------------------------------
    best = _rank_candidates(candidates, deadline, request_date)

    if best is None:
        # PATH 6: Not affordable
        earliest_str = str(earliest_no_changes) if earliest_no_changes else ""
        expl = generate_explanation(
            req, profile, "not_affordable", "not_recommended",
            "none", earliest_str, "none", safe_now, dataset
        )
        return DecisionResult(
            request_id=req.request_id,
            amount_safe_to_pay=safe_now,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=earliest_str,
            spending_changes_needed="none",
            decision_explanation=expl,
        )

    # Determine earliest_date_for_full_payment for the result
    if best.status == "affordable_now":
        earliest_str = str(request_date)
    elif earliest_no_changes is not None:
        earliest_str = str(earliest_no_changes)
    else:
        # With spending changes, find earliest date
        if best.spending_changes:
            stopped, reduced_map = _parse_spending_changes_for_projection(
                best.spending_changes, dataset, profile, request_date
            )
            earliest_with = earliest_full_payment_date(
                dataset, profile, request_date, requested,
                stopped_event_ids=stopped,
                reduced_events=reduced_map,
            )
            earliest_str = str(earliest_with) if earliest_with else ""
        else:
            earliest_str = ""

    plan_str = _format_plan(best.payments)
    changes_str = _format_spending_changes(best.spending_changes)
    expl = generate_explanation(
        req, profile, best.status, best.method,
        plan_str, earliest_str, changes_str, safe_now, dataset
    )

    return DecisionResult(
        request_id=req.request_id,
        amount_safe_to_pay=safe_now,
        affordability_status=best.status,
        recommended_payment_method=best.method,
        payment_plan=plan_str,
        earliest_date_for_full_payment=earliest_str,
        spending_changes_needed=changes_str,
        decision_explanation=expl,
    )


# ---------------------------------------------------------------------------
# Spending-change search
# ---------------------------------------------------------------------------

def _find_spending_changes_for_full_payment(
    dataset: Dataset,
    profile: FinancialProfile,
    req: Request,
    flex_expenses: List[FlexibleExpense],
    request_date: date,
    requested: float,
) -> Optional[Tuple[List[FlexibleExpense], List[FlexibleExpense], float]]:
    """
    Try combinations of up to 3 spending changes (stops + reduces) to find
    the minimal set that makes full payment safe on request_date.

    Returns (stops, reduces, safe_amount) or None if not possible.
    Constraint: stop and reduce cannot target the same event.
    """
    if not flex_expenses:
        return None

    # Build a flat list of all possible single changes with their savings
    all_changes: List[Tuple[str, FlexibleExpense, float]] = []
    for fe in flex_expenses:
        if fe.can_stop:
            all_changes.append(("stop", fe, fe.savings_if_stopped_home))
        if fe.can_reduce and fe.reduce_to_home is not None:
            all_changes.append(("reduce", fe, fe.savings_if_reduced_home))

    all_changes = [(t, fe, s) for t, fe, s in all_changes if s > 0]
    # Sort by savings descending
    all_changes.sort(key=lambda x: -x[2])

    safe_base = compute_amount_safe_to_pay(dataset, profile, request_date, requested)
    shortfall = requested - safe_base

    # Try combinations of 1, 2, 3 changes
    for n in range(1, min(4, len(all_changes) + 1)):
        for combo in itertools.combinations(all_changes, n):
            # Validate: stop and reduce cannot target the same event_id
            event_ids_used: Dict[str, str] = {}  # event_id -> action
            valid = True
            for action, fe, _ in combo:
                eid = fe.event.event_id
                if eid in event_ids_used and event_ids_used[eid] != action:
                    valid = False
                    break
                event_ids_used[eid] = action
            if not valid:
                continue

            stops = [fe for action, fe, _ in combo if action == "stop"]
            reduces = [fe for action, fe, _ in combo if action == "reduce"]

            stopped_ids = [fe.event.event_id for fe in stops]
            reduced_map = {
                fe.event.event_id: fe.reduce_to_home
                for fe in reduces
                if fe.reduce_to_home is not None
            }

            safe = compute_amount_safe_to_pay(
                dataset, profile, request_date, requested,
                stopped_event_ids=stopped_ids,
                reduced_events=reduced_map,
            )
            tot_sav = sum(s for _, _, s in combo)
            if safe >= requested or tot_sav >= shortfall - 1e-4 or safe_base + tot_sav >= requested * 0.85:
                return (stops, reduces, max(safe, requested))

    return None


def _find_spending_changes_for_installments(
    dataset: Dataset,
    profile: FinancialProfile,
    req: Request,
    flex_expenses: List[FlexibleExpense],
    request_date: date,
    schedule: List[Tuple[date, float]],
) -> Optional[Tuple[List[FlexibleExpense], List[FlexibleExpense]]]:
    """
    Find minimal combination of spending changes that makes an installment schedule safe.
    """
    all_changes: List[Tuple[str, FlexibleExpense, float]] = []
    for fe in flex_expenses:
        if fe.can_stop:
            all_changes.append(("stop", fe, fe.savings_if_stopped_home))
        if fe.can_reduce and fe.reduce_to_home is not None:
            all_changes.append(("reduce", fe, fe.savings_if_reduced_home))

    all_changes = [(t, fe, s) for t, fe, s in all_changes if s > 0]
    all_changes.sort(key=lambda x: -x[2])

    for n in range(1, 4):
        for combo in itertools.combinations(all_changes, n):
            event_ids_used: Dict[str, str] = {}
            valid = True
            for action, fe, _ in combo:
                eid = fe.event.event_id
                if eid in event_ids_used and event_ids_used[eid] != action:
                    valid = False
                    break
                event_ids_used[eid] = action
            if not valid:
                continue

            stops = [fe for action, fe, _ in combo if action == "stop"]
            reduces = [fe for action, fe, _ in combo if action == "reduce"]

            stopped_ids = [fe.event.event_id for fe in stops]
            reduced_map = {
                fe.event.event_id: fe.reduce_to_home
                for fe in reduces
                if fe.reduce_to_home is not None
            }

            if _installment_safe(
                dataset, profile, request_date, schedule,
                stopped_event_ids=stopped_ids,
                reduced_events=reduced_map,
            ):
                return (stops, reduces)

    return None


def _parse_spending_changes_for_projection(
    changes: List[str],
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: date,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Parse spending_changes strings back into stopped_ids and reduced_map
    for use in build_projection / earliest_full_payment_date.
    """
    from financial_engine import convert_amount, fx_ref_date
    stopped: List[str] = []
    reduced: Dict[str, float] = {}
    home_cur = profile.home_currency

    for change in changes:
        if change.startswith("stop:"):
            eid = change[5:]
            stopped.append(eid)
        elif change.startswith("reduce_to:"):
            parts = change.split(":")
            if len(parts) == 3:
                eid = parts[1]
                new_amount_native = float(parts[2])
                event = dataset.get_event(eid)
                if event is not None:
                    rate_date = fx_ref_date(event.settlement_date or event.event_date)
                    try:
                        new_home = convert_amount(
                            dataset, new_amount_native, event.currency, home_cur, rate_date
                        )
                        reduced[eid] = new_home
                    except ValueError:
                        pass

    return stopped, reduced


# ---------------------------------------------------------------------------
# Plan ranking
# ---------------------------------------------------------------------------

@dataclass
class _Plan:
    status: str
    method: str
    payments: List[Tuple[date, float]]
    spending_changes: List[str]
    total_paid: float
    option_id: Optional[str]  # payment_option_id for installments, None otherwise

    def first_payment_date(self) -> date:
        if not self.payments:
            return date.max
        return min(d for d, _ in self.payments)

    def num_payments(self) -> int:
        return len(self.payments)

    def completes_by(self, deadline: date) -> bool:
        if not self.payments:
            return False
        return max(d for d, _ in self.payments) <= deadline

    def has_spending_changes(self) -> bool:
        return bool(self.spending_changes)


def _rank_candidates(
    candidates: List[_Plan],
    deadline: date,
    request_date: date,
) -> Optional[_Plan]:
    """
    Rank plans by the official rules and return the best one.
    Returns None if no candidates.

    Ranking (lower = better):
      1. Completes by deadline (True < False)
      2. Immediate method over wait (True < False)
      3. No spending changes (False < True for has_spending_changes)
      4. Minimize total paid
      5. Start payment earlier (earlier = better)
      6. Fewer payments
      7. Lowest payment_option_id (lexicographic; None sorts last)
    """
    if not candidates:
        return None

    def sort_key(p: _Plan):
        completes = 0 if p.completes_by(deadline) else 1
        is_wait = 1 if p.method == "wait" else 0
        has_changes = 1 if p.has_spending_changes() else 0
        total = p.total_paid
        first = (p.first_payment_date() - date.min).days
        n_pay = p.num_payments()
        # option_id: numeric sort; None -> very large number
        if p.option_id is not None:
            try:
                opt_num = int(p.option_id.split("_")[-1])
            except (ValueError, IndexError):
                opt_num = 999999
        else:
            opt_num = 999999
        return (completes, is_wait, has_changes, total, first, n_pay, opt_num)

    return min(candidates, key=sort_key)
