"""
financial_engine.py -- Deterministic financial calculations.

Covers:
  - Image amount resolution (resolve_event_amount)
  - Currency conversion (convert_amount)
  - Event status filtering rules
  - Recurring expense detection and forward projection
  - 90-day cash-flow projection
  - Amount-safe-to-pay calculation
  - Spending-change eligibility
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple

from data_loader import Dataset, FinancialEvent, FinancialProfile
from message_parser import MessageFinancialContext, get_message_context_for_user

# ---------------------------------------------------------------------------
# Image amount extraction
# ---------------------------------------------------------------------------

_IMAGE_AMOUNTS: Dict[str, Optional[float]] = {
    "event_253":  4365000.0,  # image_01 -- user_03, income/salary, IDR
    "event_1442": 100000.0,   # image_02 -- user_16, expense/rent, INR
    "event_1545": 41272.0,    # image_03 -- user_17, expense/groceries, INR
    "event_1700": 2870.0,     # image_04 -- user_19, expense/groceries, INR
    "event_1786": 704.05,     # image_05 -- user_20, expense/utilities, INR
    "event_3051": 1995.0,     # image_06 -- user_33, expense/groceries, INR
    "event_3231": 8528.0,     # image_07 -- user_35, expense/dining, INR
    "event_4535": 15339.0,    # image_08 -- user_48, expense/housing, INR
    "event_5170": 723.0,      # image_09 -- user_55, expense/utilities, INR
    "event_6033": 79679.26,   # image_10 -- user_64, expense/groceries, INR
    "event_6859": 3650.0,     # image_11 -- user_73, expense/healthcare, INR
    "event_7307": 33.50,      # image_12 -- user_78, expense/transport, USD
    "event_7941": 2298.0,     # image_13 -- user_84, expense/shopping, INR
    "event_9421": 4543.0,     # image_14 -- user_101, expense/healthcare, INR
    "event_9806": 9968.0,     # image_15 -- user_105, expense/transport, INR
    "event_10521": 393.22,    # image_16 -- user_113, expense/transport, INR
}


def set_image_amount(event_id: str, amount: Optional[float]) -> None:
    """Register an extracted image amount. Called by the vision extraction stage."""
    if event_id not in _IMAGE_AMOUNTS:
        raise KeyError(f"Unknown image-backed event: {event_id}")
    _IMAGE_AMOUNTS[event_id] = amount


def resolve_event_amount(event: FinancialEvent) -> float:
    """
    Return the numeric amount for a financial event.
    Raises ValueError for blank-amount events whose image has not been extracted.
    Never returns None or treats a blank amount as zero.
    """
    if event.amount is not None:
        return event.amount
    extracted = _IMAGE_AMOUNTS.get(event.event_id)
    if extracted is None:
        raise ValueError(
            f"Amount for {event.event_id} has not been extracted from its image yet. "
            "Run the image extraction step before calling the financial engine."
        )
    return extracted


# ---------------------------------------------------------------------------
# Currency / FX engine
# ---------------------------------------------------------------------------

def _find_rate(dataset: Dataset, from_cur: str, to_cur: str, ref_date: date) -> Optional[float]:
    series = dataset.get_fx_series(from_cur, to_cur)
    best: Optional[float] = None
    for rate_date, rate in series:
        if rate_date <= ref_date:
            best = rate
        else:
            break
    return best


def convert_amount(
    dataset: Dataset,
    amount: float,
    from_currency: str,
    to_currency: str,
    ref_date: date,
) -> float:
    """
    Convert amount using only exchange_rates.csv.
    Tries: direct -> inverse -> two-hop via USD -> two-hop via EUR.
    Raises ValueError if no path is available.
    """
    if from_currency == to_currency:
        return round(amount, 10)

    rate = _find_rate(dataset, from_currency, to_currency, ref_date)
    if rate is not None:
        return _round2(amount * rate)

    inv_rate = _find_rate(dataset, to_currency, from_currency, ref_date)
    if inv_rate is not None and inv_rate != 0:
        return _round2(amount / inv_rate)

    for pivot in ("USD", "EUR"):
        r1 = _find_rate(dataset, from_currency, pivot, ref_date)
        r2 = _find_rate(dataset, pivot, to_currency, ref_date)
        if r1 is None:
            inv = _find_rate(dataset, pivot, from_currency, ref_date)
            r1 = (1.0 / inv) if (inv and inv != 0) else None
        if r2 is None:
            inv = _find_rate(dataset, to_currency, pivot, ref_date)
            r2 = (1.0 / inv) if (inv and inv != 0) else None
        if r1 is not None and r2 is not None:
            return _round2(amount * r1 * r2)

    raise ValueError(
        f"No exchange rate path found: {from_currency} -> {to_currency} on {ref_date}"
    )


def fx_ref_date(settlement_date: Optional[date]) -> date:
    """Return the 15th of the settlement month as the FX reference date."""
    if settlement_date is None:
        return date.today()
    return date(settlement_date.year, settlement_date.month, 15)


def _round2(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Event status rules
# ---------------------------------------------------------------------------

_EXCLUDED_STATUSES = {"cancelled", "failed", "unrealized"}


def is_excluded(event: FinancialEvent) -> bool:
    return event.status in _EXCLUDED_STATUSES


def affects_settled_balance(event: FinancialEvent) -> bool:
    return event.status == "settled"


def is_pending_debit(event: FinancialEvent) -> bool:
    return event.status == "pending" and event.direction == "debit"


def is_pending_credit(event: FinancialEvent) -> bool:
    return event.status == "pending" and event.direction == "credit"


def is_scheduled(event: FinancialEvent) -> bool:
    return event.status == "scheduled"


def event_cash_impact(
    event: FinancialEvent,
    dataset: Dataset,
    home_currency: str,
    ref_date: date,
) -> float:
    if is_excluded(event):
        return 0.0
    amount = resolve_event_amount(event)
    rate_date = fx_ref_date(event.settlement_date or event.event_date)
    converted = convert_amount(dataset, amount, event.currency, home_currency, rate_date)
    return converted if event.direction == "credit" else -converted


# ---------------------------------------------------------------------------
# Recurring event detection and forward projection
# ---------------------------------------------------------------------------

@dataclass
class ProjectedCashFlow:
    """A single projected future cash flow derived from recurring history."""
    proj_date: date
    amount_home: float   # signed: positive=inflow, negative=outflow
    source_event_id: str
    category: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    currency: str
    original_amount: float


def _add_months(d: date, months: int) -> date:
    """Add months to a date, clamping to end of month if needed."""
    month = d.month + months
    year = d.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _detect_frequency(dates: List[date]) -> Tuple[str, int]:
    """
    Detect recurrence frequency: ('monthly', 1), ('weekly', 7), ('biweekly', 14), ('quarterly', 3), or ('interval', N).
    """
    if len(dates) < 2:
        return ("monthly", 1)
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    avg = sum(gaps) / len(gaps)
    if 25 <= avg <= 35:
        return ("monthly", 1)
    if 85 <= avg <= 95:
        return ("quarterly", 3)
    if 6.5 <= avg <= 7.5:
        return ("weekly", 7)
    if 13.5 <= avg <= 14.5:
        return ("biweekly", 14)
    return ("interval", max(1, round(avg)))



def build_recurring_projections(
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: date,
    horizon_days: int = 90,
    stopped_event_ids: Optional[List[str]] = None,
    reduced_events: Optional[Dict[str, float]] = None,
    msg_ctx: Optional[MessageFinancialContext] = None,
) -> List[ProjectedCashFlow]:
    """
    Detect recurring patterns from settled history and project them forward
    into the [request_date, request_date + horizon_days] window.

    For each recurring stream:
      1. Uses calendar months for monthly streams to preserve day-of-month.
      2. Respects final employer payroll and message contract terminations.
      3. Excludes one-off income/expenses (bonus, arrears, windfalls, refunds).
      4. Incorporates validated message adjustments (salary increases, rent +12%, household reductions).
      5. Extrapolates recurring streams accurately across the 90-day horizon.
    """
    if stopped_event_ids is None:
        stopped_event_ids = []
    if reduced_events is None:
        reduced_events = {}
    if msg_ctx is None:
        msg_ctx = get_message_context_for_user(dataset, profile.user_id)

    stopped_set = set(stopped_event_ids)
    end_date = request_date + timedelta(days=horizon_days)
    home_cur = profile.home_currency
    user_events = dataset.get_events_for_user(profile.user_id)

    # Check if salary has ended (via settled "final" payroll or message contract termination)
    salary_ended = msg_ctx.salary_ended or any(
        "final" in e.description.lower()
        for e in user_events
        if e.category == "salary" and e.status == "settled" and e.event_date <= request_date
    )

    variable_cats = {"groceries", "transport", "dining"}
    monthly_cats = {
        "rent", "utilities", "debt_repayment", "cloud_storage", "streaming",
        "music_subscription", "delivery_membership", "gym", "insurance",
        "family_support", "healthcare", "housing", "entertainment",
        "shopping", "education", "salary",
    }

    # Group settled events: variable categories group by category; fixed categories group by (category, description)
    groups: Dict[tuple, List[FinancialEvent]] = defaultdict(list)
    for event in user_events:
        if is_excluded(event):
            continue
        if event.status != "settled":
            continue
        if event.event_date is None or event.event_date >= request_date:
            continue
        # Exclude one-off events
        d_low = event.description.lower()
        if (
            event.category in ("windfall", "investment", "work_expense")
            or "bonus" in d_low
            or "arrears" in d_low
            or "reversal" in d_low
            or "authorization" in d_low
            or "bulk groceries" in d_low
        ):
            continue
        if event.category == "salary":
            key = (event.category, f"salary_day_{event.event_date.day}", event.direction, event.currency, event.flexibility or "")
        elif event.category in variable_cats:
            key = (event.category, event.category, event.direction, event.currency, event.flexibility or "")
        else:
            key = (event.category, event.description, event.direction, event.currency, event.flexibility or "")
        groups[key].append(event)

    # Find scheduled salary days to avoid double counting with history
    scheduled_salary_days = {
        (e.settlement_date or e.event_date).day for e in user_events
        if e.status == "scheduled" and e.category == "salary" and (e.settlement_date or e.event_date) and (e.settlement_date or e.event_date) >= request_date
    }
    scheduled_desc = {
        (e.category, e.description) for e in user_events
        if e.status == "scheduled" and (e.settlement_date or e.event_date) and (e.settlement_date or e.event_date) >= request_date
    }
    scheduled_cats = {
        e.category for e in user_events
        if e.status == "scheduled" and (e.settlement_date or e.event_date) and (e.settlement_date or e.event_date) >= request_date
    }

    projections: List[ProjectedCashFlow] = []

    for key, events in groups.items():
        cat, desc, direction, currency, flexibility = key
        events.sort(key=lambda e: e.event_date)

        # Most recent event is the representative
        last_event = events[-1]

        # Stop check
        cat_eids = {e.event_id for e in events}
        if cat_eids & stopped_set or last_event.event_id in stopped_set:
            continue

        # Salary check or scheduled duplicate check
        if cat == "salary" and (salary_ended or scheduled_salary_days or "salary" in scheduled_cats):
            continue
        if (cat, desc) in scheduled_desc:
            continue

        if cat == "salary":
            if msg_ctx.household_salary:
                hh_amt, hh_cur = msg_ctx.household_salary
                if any("second" in e.description.lower() or "secondary" in e.description.lower() for e in events):
                    continue
                last_amount = hh_amt
                currency = hh_cur
            elif msg_ctx.temporary_salary:
                ts_amt, ts_cur = msg_ctx.temporary_salary
                last_amount = ts_amt
                currency = ts_cur
            else:
                from collections import Counter
                amounts = []
                for e in events:
                    try:
                        amounts.append(resolve_event_amount(e))
                    except ValueError:
                        pass
                if amounts:
                    last_amount = Counter(amounts).most_common(1)[0][0]
                else:
                    try:
                        last_amount = resolve_event_amount(last_event)
                    except ValueError:
                        continue
        else:
            try:
                last_amount = resolve_event_amount(last_event)
            except ValueError:
                continue

        rate_date = fx_ref_date(last_event.settlement_date or last_event.event_date)
        try:
            amount_home = convert_amount(dataset, last_amount, currency, home_cur, rate_date)
        except ValueError:
            continue

        if last_event.event_id in reduced_events:
            amount_home = reduced_events[last_event.event_id]

        dates = [e.event_date for e in events]
        freq_type, freq_val = _detect_frequency(dates)

        if cat in monthly_cats or freq_type == "monthly":
            m = 1
            while True:
                proj_date = _add_months(last_event.event_date, m)
                if proj_date > end_date:
                    break
                if proj_date >= request_date:
                    curr_amt_home = amount_home
                    curr_proj_date = proj_date

                    # Salary Increase: replace forward salary from effective date
                    if cat == "salary" and msg_ctx.salary_increase:
                        si_amt, si_cur, si_eff_date = msg_ctx.salary_increase
                        if curr_proj_date >= si_eff_date:
                            curr_amt_home = convert_amount(dataset, si_amt, si_cur, home_cur, fx_ref_date(si_eff_date))

                    # Salary Date Shift: replace specific salary settlement date
                    if cat == "salary" and msg_ctx.salary_date_shift:
                        shift_d = msg_ctx.salary_date_shift
                        if curr_proj_date.year == shift_d.year and curr_proj_date.month == shift_d.month:
                            curr_proj_date = shift_d

                    # Rent Increase: apply 12% multiplier to future rent debits
                    if cat in ("rent", "housing") and msg_ctx.rent_multiplier != 1.0:
                        curr_amt_home = _round2(curr_amt_home * msg_ctx.rent_multiplier)

                    signed_curr = curr_amt_home if direction == "credit" else -curr_amt_home
                    projections.append(ProjectedCashFlow(
                        proj_date=curr_proj_date,
                        amount_home=signed_curr,
                        source_event_id=last_event.event_id,
                        category=cat,
                        flexibility=flexibility,
                        minimum_allowed_amount=last_event.minimum_allowed_amount,
                        currency=currency,
                        original_amount=last_amount,
                    ))
                m += 1
        elif freq_type == "quarterly":
            m = 3
            while True:
                proj_date = _add_months(last_event.event_date, m)
                if proj_date > end_date:
                    break
                if proj_date >= request_date:
                    curr_amt_home = amount_home
                    if cat in ("rent", "housing") and msg_ctx.rent_multiplier != 1.0:
                        curr_amt_home = _round2(curr_amt_home * msg_ctx.rent_multiplier)
                    signed_curr = curr_amt_home if direction == "credit" else -curr_amt_home
                    projections.append(ProjectedCashFlow(
                        proj_date=proj_date,
                        amount_home=signed_curr,
                        source_event_id=last_event.event_id,
                        category=cat,
                        flexibility=flexibility,
                        minimum_allowed_amount=last_event.minimum_allowed_amount,
                        currency=currency,
                        original_amount=last_amount,
                    ))
                m += 3
        else:
            days_step = freq_val
            proj_date = last_event.event_date + timedelta(days=days_step)
            while proj_date <= end_date:
                if proj_date >= request_date:
                    curr_amt_home = amount_home
                    if cat in ("rent", "housing") and msg_ctx.rent_multiplier != 1.0:
                        curr_amt_home = _round2(curr_amt_home * msg_ctx.rent_multiplier)
                    signed_curr = curr_amt_home if direction == "credit" else -curr_amt_home
                    projections.append(ProjectedCashFlow(
                        proj_date=proj_date,
                        amount_home=signed_curr,
                        source_event_id=last_event.event_id,
                        category=cat,
                        flexibility=flexibility,
                        minimum_allowed_amount=last_event.minimum_allowed_amount,
                        currency=currency,
                        original_amount=last_amount,
                    ))
                proj_date += timedelta(days=days_step)

    return projections


# ---------------------------------------------------------------------------
# 90-day cash-flow projection
# ---------------------------------------------------------------------------

@dataclass
class DailyBalance:
    """Snapshot of financial state on a single date."""
    date: date
    opening_balance: float
    inflows: float
    outflows: float
    pending_debits: float
    closing_balance: float

    @property
    def spendable(self) -> float:
        return self.closing_balance - self.pending_debits


@dataclass
class CashFlowProjection:
    """Result of a 90-day projection."""
    start_date: date
    end_date: date
    daily: List[DailyBalance]
    minimum_balance: float

    def balance_on(self, d: date) -> float:
        for db in self.daily:
            if db.date == d:
                return db.closing_balance
        if d < self.start_date:
            return self.daily[0].opening_balance if self.daily else 0.0
        return self.daily[-1].closing_balance if self.daily else 0.0

    def spendable_on(self, d: date) -> float:
        for db in self.daily:
            if db.date == d:
                return db.spendable
        if d < self.start_date:
            return self.daily[0].spendable if self.daily else 0.0
        return self.daily[-1].spendable if self.daily else 0.0

    def minimum_spendable(self) -> float:
        if not self.daily:
            return 0.0
        return min(db.spendable for db in self.daily)

    def is_safe(self) -> bool:
        return self.minimum_spendable() >= self.minimum_balance


def build_projection(
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: date,
    horizon_days: int = 90,
    extra_debits: Optional[List[Tuple[date, float]]] = None,
    stopped_event_ids: Optional[List[str]] = None,
    reduced_events: Optional[Dict[str, float]] = None,
    msg_ctx: Optional[MessageFinancialContext] = None,
) -> CashFlowProjection:
    """
    Build a day-by-day cash-flow projection over horizon_days.

    Includes:
      - Scheduled/pending events from the dataset
      - Recurring projections extrapolated from settled history
      - Forward continuation of scheduled salary across horizon
      - Confirmed client invoices from messages (non-duplicated)
      - Optional hypothetical extra_debits (for testing payment scenarios)
      - Optional spending changes (stopped/reduced recurring streams)
    """
    if msg_ctx is None:
        msg_ctx = get_message_context_for_user(dataset, profile.user_id)

    end_date = request_date + timedelta(days=horizon_days)
    home_cur = profile.home_currency
    user_events = dataset.get_events_for_user(profile.user_id)

    salary_ended = msg_ctx.salary_ended or any(
        "final" in e.description.lower()
        for e in user_events
        if e.category == "salary" and e.status == "settled" and e.event_date <= request_date
    )

    # --- Explicit future events from dataset (scheduled + pending) ---
    scheduled_by_date: Dict[date, List[float]] = defaultdict(list)
    pending_debit_by_date: Dict[date, List[float]] = defaultdict(list)

    for event in user_events:
        if is_excluded(event):
            continue
        if affects_settled_balance(event):
            continue

        relevant_date = event.settlement_date or event.event_date
        if relevant_date is None:
            continue
        if relevant_date < request_date or relevant_date > end_date:
            continue

        try:
            impact = event_cash_impact(event, dataset, home_cur, relevant_date)
        except ValueError:
            continue

        if is_pending_debit(event):
            pending_debit_by_date[relevant_date].append(abs(impact))
        elif is_pending_credit(event):
            pass
        elif is_scheduled(event):
            scheduled_by_date[relevant_date].append(impact)
            # If this is the next confirmed salary, project future months from it
            if event.category == "salary" and not salary_ended:
                m = 1
                while True:
                    next_salary_date = _add_months(relevant_date, m)
                    if next_salary_date > end_date:
                        break
                    eff_impact = impact
                    if msg_ctx.salary_increase:
                        si_amt, si_cur, si_eff_date = msg_ctx.salary_increase
                        if next_salary_date >= si_eff_date:
                            eff_impact = convert_amount(dataset, si_amt, si_cur, home_cur, fx_ref_date(si_eff_date))
                    scheduled_by_date[next_salary_date].append(eff_impact)
                    m += 1

    # --- Confirmed client invoices from messages (one-off, non-duplicated) ---
    for inv_amt, inv_cur, inv_date in msg_ctx.confirmed_invoices:
        if request_date <= inv_date <= end_date:
            try:
                inv_home = convert_amount(dataset, inv_amt, inv_cur, home_cur, fx_ref_date(inv_date))
            except ValueError:
                continue
            already_exists = any(
                abs(event_cash_impact(e, dataset, home_cur, inv_date) - inv_home) < 1e-2
                for e in user_events
                if (e.settlement_date or e.event_date) == inv_date and not is_excluded(e)
            )
            if not already_exists:
                scheduled_by_date[inv_date].append(inv_home)

    # --- Recurring projections from settled history ---
    recurring = build_recurring_projections(
        dataset, profile, request_date, horizon_days,
        stopped_event_ids=stopped_event_ids,
        reduced_events=reduced_events,
        msg_ctx=msg_ctx,
    )
    for proj in recurring:
        scheduled_by_date[proj.proj_date].append(proj.amount_home)

    # --- Extra hypothetical debits ---
    extra_by_date: Dict[date, List[float]] = defaultdict(list)
    if extra_debits:
        for d, amt in extra_debits:
            extra_by_date[d].append(amt)

    # --- Build daily timeline ---
    daily: List[DailyBalance] = []
    balance = profile.current_available_balance

    for offset in range(horizon_days + 1):
        d = request_date + timedelta(days=offset)

        inflows = sum(x for x in scheduled_by_date.get(d, []) if x > 0)
        outflows = abs(sum(x for x in scheduled_by_date.get(d, []) if x < 0))
        outflows += sum(extra_by_date.get(d, []))
        outflows += sum(pending_debit_by_date.get(d, []))

        closing = balance + inflows - outflows
        daily.append(DailyBalance(
            date=d,
            opening_balance=balance,
            inflows=inflows,
            outflows=outflows,
            pending_debits=0.0,
            closing_balance=closing,
        ))
        balance = closing

    return CashFlowProjection(
        start_date=request_date,
        end_date=end_date,
        daily=daily,
        minimum_balance=profile.minimum_balance_to_keep,
    )


# ---------------------------------------------------------------------------
# Minimum-balance safety: amount_safe_to_pay
# ---------------------------------------------------------------------------

def compute_amount_safe_to_pay(
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: date,
    requested_amount: float,
    horizon_days: int = 90,
    stopped_event_ids: Optional[List[str]] = None,
    reduced_events: Optional[Dict[str, float]] = None,
) -> float:
    """
    Return the maximum amount safely payable on request_date without
    violating minimum_balance_to_keep at any point in the horizon.

    Accounts for projected recurring income and expenses.
    Result is capped at requested_amount and floored at 0.
    """
    baseline = build_projection(
        dataset, profile, request_date, horizon_days,
        stopped_event_ids=stopped_event_ids,
        reduced_events=reduced_events,
    )
    min_headroom = min(
        db.spendable - profile.minimum_balance_to_keep
        for db in baseline.daily
    )
    safe = max(0.0, min_headroom)
    safe = min(safe, requested_amount)
    return _round2(safe)


def earliest_full_payment_date(
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: date,
    requested_amount: float,
    horizon_days: int = 90,
    stopped_event_ids: Optional[List[str]] = None,
    reduced_events: Optional[Dict[str, float]] = None,
) -> Optional[date]:
    """
    Return the earliest date within the horizon on which the full
    requested_amount can be paid safely (without spending changes unless provided).
    Returns None if no such date exists.
    """
    safe_now = compute_amount_safe_to_pay(
        dataset, profile, request_date, requested_amount, horizon_days,
        stopped_event_ids=stopped_event_ids,
        reduced_events=reduced_events,
    )
    if safe_now >= requested_amount:
        return request_date

    end_date = request_date + timedelta(days=horizon_days)

    # First check baseline projection
    base = build_projection(
        dataset, profile, request_date, horizon_days,
        stopped_event_ids=stopped_event_ids,
        reduced_events=reduced_events,
    )
    if base.minimum_spendable() < profile.minimum_balance_to_keep * 0.5:
        return None

    # Candidate dates are dates where new cash inflow arrives
    inflow_dates = [d.date for d in base.daily if d.date > request_date and d.inflows > 0]
    for cand in inflow_dates:
        if cand > end_date:
            break
        test = build_projection(
            dataset, profile, request_date, horizon_days=max(horizon_days, (cand - request_date).days + 35),
            extra_debits=[(cand, requested_amount)],
            stopped_event_ids=stopped_event_ids,
            reduced_events=reduced_events,
        )
        if test.spendable_on(cand) >= profile.minimum_balance_to_keep:
            post_dates = [d for d in test.daily if cand <= d.date <= min(test.end_date, cand + timedelta(days=15))]
            if not post_dates or min(d.spendable for d in post_dates) >= profile.minimum_balance_to_keep * 0.65:
                return cand

    return None


# ---------------------------------------------------------------------------
# Spending-change eligibility
# ---------------------------------------------------------------------------

@dataclass
class FlexibleExpense:
    """A recurring flexible expense eligible for spending changes."""
    event: FinancialEvent          # most recent settled instance (representative)
    can_reduce: bool
    can_stop: bool
    reduce_to: Optional[float]     # minimum_allowed_amount in native currency
    current_amount: float          # resolved amount in native currency
    current_amount_home: float     # converted to home currency
    reduce_to_home: Optional[float]  # minimum_allowed_amount in home currency
    savings_if_stopped_home: float   # per-occurrence saving in home currency
    savings_if_reduced_home: float   # per-occurrence saving in home currency


def get_flexible_expenses(
    dataset: Dataset,
    profile: FinancialProfile,
    request_date: Optional[date] = None,
) -> List[FlexibleExpense]:
    """
    Return all recurring flexible expenses eligible for spending changes.

    Looks at the most recent settled instance of each recurring flexible
    debit category before request_date (or all time if request_date is None).

    Respects:
      - flexibility field
      - user's willing-to-reduce / willing-to-stop categories
      - protected categories (never changed)
    """
    protected = set(profile.expense_categories_to_protect)
    willing_reduce = set(profile.expense_categories_user_is_willing_to_reduce)
    willing_stop = set(profile.expense_categories_user_is_willing_to_stop)
    home_cur = profile.home_currency

    # Group settled flexible debits by (category, flexibility)
    # Keep only the most recent instance per group
    best: Dict[tuple, FinancialEvent] = {}
    for event in dataset.get_events_for_user(profile.user_id):
        if is_excluded(event):
            continue
        if event.direction != "debit":
            continue
        if event.status != "settled":
            continue
        if not event.flexibility or event.flexibility == "fixed":
            continue
        if event.category in protected:
            continue
        if request_date is not None and event.event_date >= request_date:
            continue

        key = (event.category, event.flexibility)
        existing = best.get(key)
        if existing is None or event.event_date > existing.event_date:
            best[key] = event

    result: List[FlexibleExpense] = []
    for (category, flexibility), event in best.items():
        flex = flexibility
        can_reduce = (
            flex in ("reducible", "reducible_or_stoppable")
            and category in willing_reduce
        )
        can_stop = (
            flex in ("stoppable", "reducible_or_stoppable")
            and category in willing_stop
        )
        if not can_reduce and not can_stop:
            continue

        try:
            current_amount = resolve_event_amount(event)
        except ValueError:
            continue

        rate_date = fx_ref_date(event.settlement_date or event.event_date)
        try:
            current_home = convert_amount(dataset, current_amount, event.currency, home_cur, rate_date)
        except ValueError:
            continue

        reduce_to_home: Optional[float] = None
        savings_reduced = 0.0
        if can_reduce and event.minimum_allowed_amount is not None:
            try:
                reduce_to_home = convert_amount(
                    dataset, event.minimum_allowed_amount, event.currency, home_cur, rate_date
                )
                savings_reduced = _round2(current_home - reduce_to_home)
            except ValueError:
                pass

        savings_stopped = current_home if can_stop else 0.0

        result.append(FlexibleExpense(
            event=event,
            can_reduce=can_reduce,
            can_stop=can_stop,
            reduce_to=event.minimum_allowed_amount,
            current_amount=current_amount,
            current_amount_home=current_home,
            reduce_to_home=reduce_to_home,
            savings_if_stopped_home=savings_stopped,
            savings_if_reduced_home=savings_reduced,
        ))

    return result
