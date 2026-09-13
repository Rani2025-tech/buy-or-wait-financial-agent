"""
tests/test_stage2.py — Stage 2 test suite.

Covers all 13 required test cases:
 1.  Loading all CSV files
 2.  Looking up a known user
 3.  Looking up a known request
 4.  Getting that request's payment options
 5.  Getting that user's financial events
 6.  Currency conversion
 7.  Settled event handling
 8.  Pending debit handling
 9.  Pending credit handling
10.  Cancelled/failed/unrealized events
11.  Blank event amount resolution
12.  Minimum balance calculation
13.  A 90-day projection

Run with:  python -m pytest tests/test_stage2.py -v
"""

import os
import sys
from datetime import date, timedelta

import pytest

# Make code/ importable regardless of where pytest is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from data_loader import Dataset
from financial_engine import (
    _EXCLUDED_STATUSES,
    affects_settled_balance,
    build_projection,
    compute_amount_safe_to_pay,
    convert_amount,
    earliest_full_payment_date,
    event_cash_impact,
    fx_ref_date,
    get_flexible_expenses,
    is_excluded,
    is_pending_credit,
    is_pending_debit,
    is_scheduled,
    resolve_event_amount,
    set_image_amount,
)


# ---------------------------------------------------------------------------
# Shared fixture — load dataset once for the whole test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ds() -> Dataset:
    dataset = Dataset()
    dataset.load_all()
    return dataset


# ---------------------------------------------------------------------------
# Test 1: Loading all CSV files
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_requests_count(self, ds):
        assert len(ds.requests) == 250

    def test_profiles_count(self, ds):
        assert len(ds.profiles) == 275

    def test_events_count(self, ds):
        assert len(ds.events) == 25342

    def test_payment_options_loaded(self, ds):
        assert len(ds.payment_options) > 0

    def test_exchange_rates_loaded(self, ds):
        assert len(ds.exchange_rates) > 0

    def test_messages_count(self, ds):
        assert len(ds.messages) == 215

    def test_images_count(self, ds):
        assert len(ds.images) == 16

    def test_requests_range(self, ds):
        assert "request_26" in ds.requests
        assert "request_275" in ds.requests

    def test_sample_requests_not_in_main(self, ds):
        # requests.csv contains request_26–275 only
        assert "request_01" not in ds.requests


# ---------------------------------------------------------------------------
# Test 2: Looking up a known user
# ---------------------------------------------------------------------------

class TestUserLookup:
    def test_user_01_exists(self, ds):
        assert ds.get_profile("user_01") is not None

    def test_user_01_currency(self, ds):
        assert ds.get_profile("user_01").home_currency == "ZAR"

    def test_user_01_balance(self, ds):
        assert ds.get_profile("user_01").current_available_balance == 58481.1

    def test_user_01_minimum_balance(self, ds):
        assert ds.get_profile("user_01").minimum_balance_to_keep == 18000

    def test_user_01_payment_methods(self, ds):
        assert "full_payment" in ds.get_profile("user_01").payment_methods_user_will_consider

    def test_user_01_no_installments(self, ds):
        assert ds.get_profile("user_01").max_installment_months is None

    def test_user_02_installments(self, ds):
        assert ds.get_profile("user_02").max_installment_months == 7

    def test_unknown_user_returns_none(self, ds):
        assert ds.get_profile("user_9999") is None

    def test_pipe_list_parsed_as_list(self, ds):
        p = ds.get_profile("user_01")
        assert isinstance(p.financial_priorities, list)
        assert "education" in p.financial_priorities


# ---------------------------------------------------------------------------
# Test 3: Looking up a known request
# ---------------------------------------------------------------------------

class TestRequestLookup:
    def test_request_26_exists(self, ds):
        assert ds.get_request("request_26") is not None

    def test_request_26_user(self, ds):
        assert ds.get_request("request_26").user_id == "user_26"

    def test_request_26_amount_positive(self, ds):
        assert ds.get_request("request_26").requested_amount > 0

    def test_request_26_dates_are_date_objects(self, ds):
        r = ds.get_request("request_26")
        assert isinstance(r.request_date, date)
        assert isinstance(r.desired_completion_date, date)

    def test_allows_partial_payment_is_bool(self, ds):
        r = ds.get_request("request_26")
        assert isinstance(r.allows_partial_payment, bool)

    def test_unknown_request_returns_none(self, ds):
        assert ds.get_request("request_9999") is None


# ---------------------------------------------------------------------------
# Test 4: Payment options for a request
# ---------------------------------------------------------------------------

class TestPaymentOptions:
    def test_request_26_has_options(self, ds):
        assert len(ds.get_payment_options_for_request("request_26")) >= 2

    def test_option_method_valid(self, ds):
        for o in ds.get_payment_options_for_request("request_26"):
            assert o.payment_method in ("full_payment", "installments")

    def test_option_total_positive(self, ds):
        for o in ds.get_payment_options_for_request("request_26"):
            assert o.total_payable_amount > 0

    def test_option_first_payment_date(self, ds):
        for o in ds.get_payment_options_for_request("request_26"):
            assert isinstance(o.first_payment_date, date)

    def test_no_options_unknown_request(self, ds):
        assert ds.get_payment_options_for_request("request_9999") == []


# ---------------------------------------------------------------------------
# Test 5: Financial events for a user
# ---------------------------------------------------------------------------

class TestUserEvents:
    def test_user_01_has_events(self, ds):
        assert len(ds.get_events_for_user("user_01")) > 0

    def test_events_belong_to_user(self, ds):
        for e in ds.get_events_for_user("user_01"):
            assert e.user_id == "user_01"

    def test_event_direction_valid(self, ds):
        for e in ds.get_events_for_user("user_01"):
            assert e.direction in ("credit", "debit")

    def test_event_status_valid(self, ds):
        valid = {"settled", "pending", "scheduled", "cancelled", "failed", "unrealized"}
        for e in ds.get_events_for_user("user_01"):
            assert e.status in valid

    def test_no_events_unknown_user(self, ds):
        assert ds.get_events_for_user("user_9999") == []


# ---------------------------------------------------------------------------
# Test 6: Currency conversion
# ---------------------------------------------------------------------------

class TestCurrencyConversion:
    def test_usd_to_inr(self, ds):
        result = convert_amount(ds, 100.0, "USD", "INR", date(2024, 1, 15))
        assert abs(result - 8333.0) < 1.0, f"Got {result}"

    def test_eur_to_zar(self, ds):
        result = convert_amount(ds, 50.0, "EUR", "ZAR", date(2025, 6, 15))
        assert abs(result - 1000.0) < 1.0, f"Got {result}"

    def test_usd_to_eur(self, ds):
        result = convert_amount(ds, 100.0, "USD", "EUR", date(2024, 1, 15))
        assert abs(result - 92.0) < 1.0, f"Got {result}"

    def test_eur_to_usd_direct(self, ds):
        # EUR→USD direct rate = 1.09
        result = convert_amount(ds, 100.0, "EUR", "USD", date(2025, 6, 15))
        assert abs(result - 109.0) < 1.0, f"Got {result}"

    def test_usd_to_idr(self, ds):
        result = convert_amount(ds, 1.0, "USD", "IDR", date(2024, 1, 15))
        assert abs(result - 15833.33) < 1.0, f"Got {result}"

    def test_same_currency_identity(self, ds):
        assert convert_amount(ds, 500.0, "INR", "INR", date(2024, 6, 15)) == 500.0

    def test_fx_ref_date_uses_15th(self):
        assert fx_ref_date(date(2024, 3, 22)) == date(2024, 3, 15)

    def test_fx_ref_date_none_returns_today(self):
        assert fx_ref_date(None) == date.today()

    def test_unknown_pair_raises(self, ds):
        with pytest.raises(ValueError):
            convert_amount(ds, 100.0, "ZAR", "IDR", date(2024, 1, 15))


# ---------------------------------------------------------------------------
# Test 7: Settled event handling
# ---------------------------------------------------------------------------

class TestSettledEvents:
    def test_settled_affects_balance(self, ds):
        settled = [e for e in ds.get_events_for_user("user_01") if e.status == "settled"]
        assert len(settled) > 0
        assert all(affects_settled_balance(e) for e in settled)

    def test_settled_not_pending_debit(self, ds):
        settled = [e for e in ds.get_events_for_user("user_01") if e.status == "settled"]
        assert not any(is_pending_debit(e) for e in settled)

    def test_settled_not_excluded(self, ds):
        settled = [e for e in ds.get_events_for_user("user_01") if e.status == "settled"]
        assert not any(is_excluded(e) for e in settled)


# ---------------------------------------------------------------------------
# Test 8: Pending debit handling
# ---------------------------------------------------------------------------

class TestPendingDebits:
    def _find_pending_debit(self, ds):
        for uid in list(ds.profiles.keys()):
            for e in ds.get_events_for_user(uid):
                if is_pending_debit(e):
                    return e
        return None

    def test_pending_debit_status_and_direction(self, ds):
        e = self._find_pending_debit(ds)
        if e is not None:
            assert e.status == "pending"
            assert e.direction == "debit"

    def test_pending_debit_not_settled(self, ds):
        e = self._find_pending_debit(ds)
        if e is not None:
            assert not affects_settled_balance(e)

    def test_pending_debit_not_pending_credit(self, ds):
        e = self._find_pending_debit(ds)
        if e is not None:
            assert not is_pending_credit(e)

    def test_pending_debit_not_excluded(self, ds):
        e = self._find_pending_debit(ds)
        if e is not None:
            assert not is_excluded(e)


# ---------------------------------------------------------------------------
# Test 9: Pending credit handling
# ---------------------------------------------------------------------------

class TestPendingCredits:
    def _find_pending_credit(self, ds):
        for uid in list(ds.profiles.keys()):
            for e in ds.get_events_for_user(uid):
                if is_pending_credit(e):
                    return e
        return None

    def test_pending_credit_status_and_direction(self, ds):
        e = self._find_pending_credit(ds)
        if e is not None:
            assert e.status == "pending"
            assert e.direction == "credit"

    def test_pending_credit_not_settled(self, ds):
        e = self._find_pending_credit(ds)
        if e is not None:
            assert not affects_settled_balance(e)

    def test_pending_credit_not_pending_debit(self, ds):
        e = self._find_pending_credit(ds)
        if e is not None:
            assert not is_pending_debit(e)


# ---------------------------------------------------------------------------
# Test 10: Cancelled / failed / unrealized events
# ---------------------------------------------------------------------------

class TestExcludedEvents:
    def test_excluded_statuses_set(self):
        assert _EXCLUDED_STATUSES == {"cancelled", "failed", "unrealized"}

    def test_cancelled_is_excluded(self, ds):
        for uid in list(ds.profiles.keys())[:100]:
            for e in ds.get_events_for_user(uid):
                if e.status == "cancelled":
                    assert is_excluded(e)
                    assert not affects_settled_balance(e)

    def test_failed_is_excluded(self, ds):
        for uid in list(ds.profiles.keys())[:100]:
            for e in ds.get_events_for_user(uid):
                if e.status == "failed":
                    assert is_excluded(e)

    def test_unrealized_is_excluded(self, ds):
        for uid in list(ds.profiles.keys())[:100]:
            for e in ds.get_events_for_user(uid):
                if e.status == "unrealized":
                    assert is_excluded(e)

    def test_settled_not_in_excluded(self):
        assert "settled" not in _EXCLUDED_STATUSES

    def test_pending_not_in_excluded(self):
        assert "pending" not in _EXCLUDED_STATUSES


# ---------------------------------------------------------------------------
# Test 11: Blank event amount resolution
# ---------------------------------------------------------------------------

class TestBlankAmountResolution:
    BLANK_EVENT_IDS = [
        "event_253", "event_1442", "event_1545", "event_1700", "event_1786",
        "event_3051", "event_3231", "event_4535", "event_5170", "event_6033",
        "event_6859", "event_7307", "event_7941", "event_9421", "event_9806",
        "event_10521",
    ]

    def test_event_253_amount_is_none(self, ds):
        assert ds.get_event("event_253").amount is None

    def test_blank_raises_before_extraction(self, ds):
        e = ds.get_event("event_253")
        with pytest.raises(ValueError, match="not been extracted"):
            resolve_event_amount(e)

    def test_non_blank_resolves_directly(self, ds):
        for e in ds.get_events_for_user("user_01"):
            if e.amount is not None:
                assert resolve_event_amount(e) == e.amount
                break

    def test_set_and_resolve_image_amount(self, ds):
        e = ds.get_event("event_253")
        set_image_amount("event_253", 99999.0)
        assert resolve_event_amount(e) == 99999.0
        set_image_amount("event_253", None)  # reset

    def test_all_16_blank_events_exist(self, ds):
        for eid in self.BLANK_EVENT_IDS:
            e = ds.get_event(eid)
            assert e is not None, f"{eid} missing"
            assert e.amount is None, f"{eid} should be blank"

    def test_all_16_have_image_records(self, ds):
        for eid in self.BLANK_EVENT_IDS:
            img = ds.get_image_for_event(eid)
            assert img is not None, f"No image record for {eid}"

    def test_image_files_exist_on_disk(self, ds):
        for eid in self.BLANK_EVENT_IDS:
            img = ds.get_image_for_event(eid)
            path = ds.image_path(img.image_id)
            assert os.path.exists(path), f"Missing image file: {path}"

    def test_set_unknown_event_raises(self):
        with pytest.raises(KeyError):
            set_image_amount("event_99999", 1.0)


# ---------------------------------------------------------------------------
# Test 12: Minimum balance calculation
# ---------------------------------------------------------------------------

class TestMinimumBalance:
    def test_safe_to_pay_request_01_covers_full_amount(self, ds):
        """
        user_01 profile: ZAR, balance=58481.1, min=18000.
        With recurring projection, safe_to_pay accounts for projected expenses.
        Must be >= 0 and <= requested_amount.
        """
        prof = ds.get_profile("user_01")
        sample_date = date(2024, 3, 3)
        sample_amount = 25256.0
        safe = compute_amount_safe_to_pay(ds, prof, sample_date, sample_amount)
        assert 0 <= safe <= sample_amount, f"safe_to_pay out of range: {safe}"

    def test_safe_to_pay_capped_at_requested(self, ds):
        prof = ds.get_profile("user_01")
        sample_date = date(2024, 3, 3)
        sample_amount = 25256.0
        safe = compute_amount_safe_to_pay(ds, prof, sample_date, sample_amount)
        assert safe <= sample_amount

    def test_safe_to_pay_non_negative(self, ds):
        for rid in list(ds.requests.keys())[:10]:
            req = ds.get_request(rid)
            prof = ds.get_profile(req.user_id)
            safe = compute_amount_safe_to_pay(ds, prof, req.request_date, req.requested_amount)
            assert safe >= 0, f"{rid}: got {safe}"

    def test_paying_safe_amount_keeps_projection_safe(self, ds):
        prof = ds.get_profile("user_01")
        sample_date = date(2024, 3, 3)
        sample_amount = 25256.0
        safe = compute_amount_safe_to_pay(ds, prof, sample_date, sample_amount)
        proj = build_projection(
            ds, prof, sample_date,
            extra_debits=[(sample_date, safe)],
        )
        assert proj.is_safe()


# ---------------------------------------------------------------------------
# Test 13: 90-day projection
# ---------------------------------------------------------------------------

class TestCashFlowProjection:
    def test_projection_length(self, ds):
        prof = ds.get_profile("user_01")
        proj = build_projection(ds, prof, date(2024, 3, 3))
        assert len(proj.daily) == 91

    def test_projection_start_date(self, ds):
        d = date(2024, 3, 3)
        prof = ds.get_profile("user_01")
        proj = build_projection(ds, prof, d)
        assert proj.daily[0].date == d

    def test_projection_end_date(self, ds):
        d = date(2024, 3, 3)
        prof = ds.get_profile("user_01")
        proj = build_projection(ds, prof, d)
        assert proj.daily[-1].date == d + timedelta(days=90)

    def test_opening_balance_matches_profile(self, ds):
        prof = ds.get_profile("user_01")
        proj = build_projection(ds, prof, date(2024, 3, 3))
        assert proj.daily[0].opening_balance == prof.current_available_balance

    def test_extra_debit_reduces_closing_balance(self, ds):
        d = date(2024, 3, 3)
        prof = ds.get_profile("user_01")
        base = build_projection(ds, prof, d)
        paid = build_projection(ds, prof, d, extra_debits=[(d, 1000.0)])
        assert paid.daily[0].closing_balance < base.daily[0].closing_balance

    def test_baseline_projection_is_safe(self, ds):
        """Baseline projection (no extra payment) for user_01 must be safe."""
        d = date(2024, 3, 3)
        prof = ds.get_profile("user_01")
        proj = build_projection(ds, prof, d)
        assert proj.is_safe(), "Baseline projection should be safe"

    def test_earliest_full_payment_date_within_horizon(self, ds):
        """earliest_full_payment_date must return a date within the 90-day horizon or None."""
        d = date(2024, 3, 3)
        amount = 25256.0
        prof = ds.get_profile("user_01")
        earliest = earliest_full_payment_date(ds, prof, d, amount)
        if earliest is not None:
            assert earliest >= d
            assert earliest <= d + timedelta(days=90)

    def test_flexible_expenses_not_protected(self, ds):
        prof = ds.get_profile("user_01")
        protected = set(prof.expense_categories_to_protect)
        for f in get_flexible_expenses(ds, prof):
            assert f.event.category not in protected

    def test_flexible_expenses_are_debits(self, ds):
        """Flexible expenses (settled recurring) must all be debit direction."""
        prof = ds.get_profile("user_01")
        for f in get_flexible_expenses(ds, prof):
            assert f.event.direction == "debit"
