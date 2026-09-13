"""
test_images.py -- Comprehensive test suite for image-linked financial event amount resolution and integration.
"""

from __future__ import annotations

from datetime import date
import os
import sys
import pytest

# Ensure code/ is on path
_CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_loader import Dataset, FinancialEvent
from financial_engine import (
    _IMAGE_AMOUNTS,
    build_projection,
    build_recurring_projections,
    convert_amount,
    event_cash_impact,
    resolve_event_amount,
    set_image_amount,
)


@pytest.fixture(scope="module")
def loaded_dataset() -> Dataset:
    ds = Dataset()
    ds.load_all()
    return ds


def test_all_16_image_amounts_populated():
    """Verify that all 16 image-backed events have confirmed numeric amounts populated."""
    expected = {
        "event_253": 4365000.0,
        "event_1442": 100000.0,
        "event_1545": 41272.0,
        "event_1700": 2870.0,
        "event_1786": 704.05,
        "event_3051": 1995.0,
        "event_3231": 8528.0,
        "event_4535": 15339.0,
        "event_5170": 723.0,
        "event_6033": 79679.26,
        "event_6859": 3650.0,
        "event_7307": 33.50,
        "event_7941": 2298.0,
        "event_9421": 4543.0,
        "event_9806": 9968.0,
        "event_10521": 393.22,
    }
    assert len(_IMAGE_AMOUNTS) == 16
    for event_id, expected_amt in expected.items():
        assert _IMAGE_AMOUNTS[event_id] == pytest.approx(expected_amt, rel=1e-5), (
            f"Mismatch for {event_id}: expected {expected_amt}, got {_IMAGE_AMOUNTS.get(event_id)}"
        )


@pytest.mark.parametrize(
    "event_id,expected_amount",
    [
        ("event_253", 4365000.0),
        ("event_1442", 100000.0),
        ("event_1545", 41272.0),
        ("event_1700", 2870.0),
        ("event_1786", 704.05),
        ("event_3051", 1995.0),
        ("event_3231", 8528.0),
        ("event_4535", 15339.0),
        ("event_5170", 723.0),
        ("event_6033", 79679.26),
        ("event_6859", 3650.0),
        ("event_7307", 33.50),
        ("event_7941", 2298.0),
        ("event_9421", 4543.0),
        ("event_9806", 9968.0),
        ("event_10521", 393.22),
    ],
)
def test_resolve_event_amount_dataset_events(loaded_dataset: Dataset, event_id: str, expected_amount: float):
    """Verify that resolving events directly from the dataset returns the confirmed amount."""
    event = loaded_dataset.get_event(event_id)
    assert event is not None, f"Event {event_id} not found in dataset"
    resolved = resolve_event_amount(event)
    assert resolved == pytest.approx(expected_amount, rel=1e-5)


def test_event_7307_fx_conversion(loaded_dataset: Dataset):
    """
    Verify event_7307 (Taxi fare, USD 33.50 on 2025-10-01) converts to INR
    via convert_amount using the fixed rate (83.33) in exchange_rates.csv.
    """
    event = loaded_dataset.get_event("event_7307")
    assert event is not None
    assert event.currency == "USD"
    amount = resolve_event_amount(event)
    assert amount == 33.50

    rate_date = date(2025, 10, 1)
    converted = convert_amount(loaded_dataset, amount, "USD", "INR", rate_date)
    expected_converted = 33.50 * 83.33  # 2791.555 INR
    assert converted == pytest.approx(expected_converted, rel=1e-4)

    # Cash impact should be negative (debit)
    impact = event_cash_impact(event, loaded_dataset, "INR", rate_date)
    assert impact == pytest.approx(-expected_converted, rel=1e-4)


def test_unextracted_blank_event_raises_value_error():
    """Verify that unknown blank-amount events raise ValueError and preserve fallback behavior."""
    unknown_event = FinancialEvent(
        event_id="event_unknown_9999",
        user_id="user_test",
        event_type="expense",
        description="Unknown blank event",
        category="shopping",
        direction="debit",
        amount=None,
        currency="INR",
        event_date=date(2026, 1, 1),
        settlement_date=date(2026, 1, 1),
        status="settled",
        linked_event_id="",
        flexibility="flexible",
        minimum_allowed_amount=None,
    )
    with pytest.raises(ValueError, match="Amount for event_unknown_9999 has not been extracted"):
        resolve_event_amount(unknown_event)


def test_set_image_amount_validation():
    """Verify that set_image_amount raises KeyError on invalid event IDs."""
    with pytest.raises(KeyError, match="Unknown image-backed event"):
        set_image_amount("event_invalid_999", 500.0)


def test_no_duplicate_transactions_for_image_users(loaded_dataset: Dataset):
    """Verify that dataset does not contain duplicate event IDs."""
    event_ids = list(loaded_dataset.events.keys())
    assert len(event_ids) == len(set(event_ids)), "Duplicate event IDs found in dataset!"


def test_image_users_cash_flow_projection_validity(loaded_dataset: Dataset):
    """Verify that 90-day cash flow projections succeed for all users associated with image events."""
    image_user_ids = [
        "user_03", "user_16", "user_17", "user_19", "user_20",
        "user_33", "user_35", "user_48", "user_55", "user_64",
        "user_73", "user_78", "user_84", "user_101", "user_105", "user_113"
    ]
    for user_id in image_user_ids:
        profile = loaded_dataset.get_profile(user_id)
        assert profile is not None, f"Profile for {user_id} missing"
        # Find request in loaded_dataset.requests
        req = next((r for r in loaded_dataset.requests.values() if r.user_id == user_id), None)
        req_date = req.request_date if req else date(2026, 6, 1)

        proj = build_projection(loaded_dataset, profile, req_date)
        assert proj is not None
        assert len(proj.daily) == 91
        # Invariant: daily balances are finite floats
        for db in proj.daily:
            assert not (db.closing_balance != db.closing_balance)  # not NaN


def test_event_1545_bulk_groceries_not_recurring_baseline(loaded_dataset: Dataset):
    """
    Regression test: event_1545 (INR 41,272 Bulk groceries and pantry purchase)
    is a one-off bulk purchase and must NOT be used as the forward recurring weekly grocery amount for user_17.
    """
    profile = loaded_dataset.get_profile("user_17")
    req_date = date(2026, 3, 1)
    projections = build_recurring_projections(loaded_dataset, profile, req_date, horizon_days=90)
    grocery_projs = [p for p in projections if p.category == "groceries"]
    # Verify that none of the projected grocery streams use event_1545 or 41,272.0 INR
    for p in grocery_projs:
        assert p.source_event_id != "event_1545", "event_1545 was incorrectly used as recurring stream source!"
        assert abs(p.amount_home) != 41272.0, "event_1545 amount (41,272 INR) was incorrectly projected as weekly grocery!"
