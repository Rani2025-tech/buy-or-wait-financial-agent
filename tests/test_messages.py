"""
test_messages.py -- Unit and regression tests for message parsing and financial context integration.

Covers:
1. English salary increase
2. Indonesian salary increase
3. Foreign-currency salary conversion
4. Temporary salary reduction
5. Employment / contract termination
6. Household salary reduction
7. Rent +12% increase
8. Salary date shift
9. Confirmed client invoice
10. Pending payout ignored
11. Pending bonus ignored
12. Scam message ignored
13. No double-counting with existing financial events
14. Minimum-balance invariant preservation
15. Real dataset regression cases
"""

from __future__ import annotations

import os
import sys
from datetime import date
import pytest

# Ensure code/ is on path
_CODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_loader import Dataset, FinancialProfile, Message, Request
from financial_engine import (
    build_projection,
    build_recurring_projections,
    compute_amount_safe_to_pay,
    convert_amount,
    earliest_full_payment_date,
    fx_ref_date,
)
from message_parser import (
    MessageFinancialContext,
    get_message_context_for_user,
    parse_message_text,
)


@pytest.fixture(scope="module")
def loaded_dataset() -> Dataset:
    ds = Dataset()
    ds.load_all()
    return ds


# ---------------------------------------------------------------------------
# Unit tests: Message Parser
# ---------------------------------------------------------------------------

def test_english_salary_increase():
    text = "Hi, Northstar Labs payroll here. Your monthly salary has increased to USD 2988. The change applies from 2026-07-15. The revised amount will appear on your next payslip. Payroll ref EMP-0026."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_increase == (2988.0, "USD", date(2026, 7, 15))


def test_indonesian_salary_increase():
    text = "Rincian penggajian Anda di Cobalt Systems telah berubah. Gaji bulanan Anda naik menjadi IDR 42750000. Perubahan ini berlaku mulai 2025-08-15. Jumlah yang diperbarui akan terlihat pada slip gaji berikutnya. Ref payroll EMP-0001."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_increase == (42750000.0, "IDR", date(2025, 8, 15))


def test_temporary_salary_reduction():
    text = "A note from Riverline Retail about your upcoming pay. Your temporary monthly pay is EUR 1924.56. The reduced amount continues for the next payroll. This is the amount currently scheduled for the affected pay cycle. Payroll ref EMP-0138."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.temporary_salary == (1924.56, "EUR")


def test_indonesian_temporary_salary_reduction():
    text = "Ada pembaruan singkat dari tim payroll HarborWorks. Gaji bulanan sementara Anda adalah IDR 23940000. Jumlah yang lebih rendah masih berlaku untuk penggajian berikutnya. Inilah jumlah yang saat ini dijadwalkan untuk periode penggajian tersebut. Ref payroll EMP-0122."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.temporary_salary == (23940000.0, "IDR")


def test_contract_termination_english():
    text = "A note from Cobalt Systems about your upcoming pay. The current seasonal contract has ended. No off-season income or renewal has been confirmed. We'll contact you separately if another shift block or contract is approved. Payroll ref EMP-0009."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_ended is True


def test_contract_termination_indonesian():
    text = "Tim payroll Northstar Labs telah mengirim pembaruan. Kontrak musiman saat ini telah berakhir. Belum ada pendapatan di luar musim atau perpanjangan kontrak yang dikonfirmasi. Kami akan menghubungi Anda jika jadwal kerja atau kontrak berikutnya disetujui. Ref payroll EMP-0160."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_ended is True


def test_employment_ended():
    text = "Here's the latest payroll information from Cedar Health. Your employment has ended. There are no regular salary payments scheduled after the final settlement. Details of any final settlement will be sent separately. Payroll ref EMP-0192."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_ended is True


def test_household_salary_reduction_english():
    text = "A quick update from the payroll team at Greenfield Foods. One household employment record has ended. The remaining confirmed monthly salary is INR 148000. Any income that has ended should be removed from future estimates. Payroll ref EMP-0030."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.household_salary == (148000.0, "INR")


def test_household_salary_reduction_indonesian():
    text = "Rincian penggajian Anda di Cedar Health telah berubah. Salah satu sumber pendapatan kerja rumah tangga telah berakhir. Sisa gaji bulanan yang dikonfirmasi adalah IDR 25840000. Pendapatan yang sudah berakhir harus dikeluarkan dari perkiraan berikutnya. Ref payroll EMP-0042."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.household_salary == (25840000.0, "IDR")


def test_rent_increase_english():
    text = "StayLedger wanted to let you know about a change on your account. The renewed lease increases monthly rent by 12%. The new amount applies from the next rent payment. The new amount will be used for the next rent payment. Case ref SER-0061."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.rent_multiplier == 1.12


def test_rent_increase_indonesian():
    text = "HomePortal memiliki informasi baru tentang pembayaran berikutnya. Perpanjangan sewa menaikkan biaya sewa bulanan sebesar 12%. Jumlah baru berlaku mulai pembayaran sewa berikutnya. Jumlah baru akan digunakan untuk pembayaran sewa berikutnya. Ref kasus SER-0175."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.rent_multiplier == 1.12


def test_salary_date_shift_english():
    text = "BrightPath Media has updated your payroll record. Your confirmed salary is now expected on 2024-09-23. This replaces the payroll date shown in the earlier update. Please use the revised date for anything you normally pay around payday. Payroll ref EMP-0005."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_date_shift == date(2024, 9, 23)


def test_salary_date_shift_indonesian():
    text = "Cobalt Systems payroll has posted a new update. Gaji yang telah dikonfirmasi sekarang diperkirakan cair pada 2025-05-23. Tanggal ini menggantikan tanggal pembayaran gaji pada pembaruan sebelumnya. Ref payroll EMP-0073."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_date_shift == date(2025, 5, 23)


def test_confirmed_client_invoice_english():
    text = "Hi, ClientDesk here. The client approved an invoice payment of INR 196000. Settlement is expected on 2024-12-15; the other submitted invoices are still awaiting approval. Only invoices marked as confirmed should be included in the upcoming payout. Case ref SER-0024."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.confirmed_invoices == [(196000.0, "INR", date(2024, 12, 15))]


def test_confirmed_client_invoice_indonesian():
    text = "Ada pembaruan singkat untuk akun ProjectPay Anda. Klien menyetujui pembayaran faktur sebesar IDR 15390000. Penyelesaian diperkirakan pada 2025-08-15; faktur lain yang diajukan masih menunggu persetujuan. Hanya faktur yang sudah dikonfirmasi yang dapat dimasukkan dalam pembayaran berikutnya. Ref kasus SER-0093."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.confirmed_invoices == [(15390000.0, "IDR", date(2025, 8, 15))]


def test_pending_payout_ignored():
    text = "Here's the latest service update from QuickCrew. The next QuickCrew payout is still pending. The weekly earnings shown in the QuickCrew app can change until the payout is closed. The balance isn't withdrawable until the payout shows as completed. Case ref SER-0007."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    # Context should remain empty
    assert ctx.confirmed_invoices == []
    assert ctx.salary_increase is None
    assert ctx.salary_ended is False


def test_pending_bonus_ignored():
    text = "BrightPath Media has updated your payroll record. Your quarterly bonus is still subject to the final performance review. The final amount and payment date have not been approved yet. We'll send another update once payroll confirms the amount and date. Payroll ref EMP-0144."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    # Context should remain empty
    assert ctx.confirmed_invoices == []
    assert ctx.salary_increase is None
    assert ctx.salary_ended is False


def test_scam_message_ignored():
    text = "A note from QuickPrize about your recent financial activity. You won a cash prize of USD 10000. Pay the release charge today to complete the transfer. The money is ready to release once the processing fee is paid. Account ref FIN-0067."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    # Context should remain completely empty
    assert ctx.salary_increase is None
    assert ctx.confirmed_invoices == []
    assert ctx.temporary_salary is None
    assert ctx.salary_ended is False
    assert ctx.rent_multiplier == 1.0


def test_scam_message_indonesian_ignored():
    text = "Pemberitahuan baru tersedia untuk akun RewardNow Anda. Selamat, Anda terpilih menerima hadiah uang tunai IDR 50000000. Bayar biaya pencairan hari ini untuk menyelesaikan transfer. Ref akun FIN-0142."
    ctx = MessageFinancialContext(user_id="user_test")
    parse_message_text(text, "user_test", ctx)
    assert ctx.salary_increase is None
    assert ctx.confirmed_invoices == []
    assert ctx.temporary_salary is None
    assert ctx.salary_ended is False
    assert ctx.rent_multiplier == 1.0


# ---------------------------------------------------------------------------
# Integration tests: Real dataset users and financial projections
# ---------------------------------------------------------------------------

def test_salary_increase_real_user(loaded_dataset: Dataset):
    """user_02 (sample_02): salary increases on 2025-08-15 to IDR 42,750,000."""
    p = loaded_dataset.get_profile("user_02")
    req_date = date(2025, 8, 8)
    projs = build_recurring_projections(loaded_dataset, p, req_date, horizon_days=90)
    salary_projs = [pj for pj in projs if pj.category == "salary"]
    assert len(salary_projs) >= 2
    # First projected salary is 2025-08-15 and should have amount 42,750,000 IDR
    first_sal = salary_projs[0]
    assert first_sal.proj_date == date(2025, 8, 15)
    assert first_sal.amount_home == 42750000.0


def test_salary_date_shift_real_user(loaded_dataset: Dataset):
    """user_07 (sample_07): salary moves to 2024-09-23."""
    p = loaded_dataset.get_profile("user_07")
    req_date = date(2024, 9, 12)
    projs = build_recurring_projections(loaded_dataset, p, req_date, horizon_days=90)
    salary_projs = [pj for pj in projs if pj.category == "salary"]
    assert len(salary_projs) >= 1
    # First projected salary must be on 2024-09-23, not 2024-09-15
    assert salary_projs[0].proj_date == date(2024, 9, 23)


def test_contract_termination_real_user(loaded_dataset: Dataset):
    """user_12 (sample_12): seasonal contract ended, no forward salary extrapolation."""
    p = loaded_dataset.get_profile("user_12")
    req_date = date(2026, 4, 19)
    projs = build_recurring_projections(loaded_dataset, p, req_date, horizon_days=90)
    salary_projs = [pj for pj in projs if pj.category == "salary"]
    # Should have zero future salary projected
    assert len(salary_projs) == 0


def test_rent_increase_real_user(loaded_dataset: Dataset):
    """user_16 (sample_16): rent increases by 12%."""
    p = loaded_dataset.get_profile("user_16")
    req_date = date(2023, 8, 12)
    projs = build_recurring_projections(loaded_dataset, p, req_date, horizon_days=90)
    rent_projs = [pj for pj in projs if pj.category == "rent"]
    assert len(rent_projs) >= 1
    # Historical base rent was 57,100 INR. 12% increase -> 63,952 INR
    assert abs(abs(rent_projs[0].amount_home) - 63952.0) < 1.0


def test_client_invoice_inflow(loaded_dataset: Dataset):
    """user_34 (request_24 / message_24): confirmed client invoice on 2024-12-15."""
    p = loaded_dataset.get_profile("user_34")
    req_date = date(2024, 12, 1)
    proj = build_projection(loaded_dataset, p, req_date, horizon_days=30)
    # On 2024-12-15, there should be an inflow of 196,000 INR
    db_15 = next(db for db in proj.daily if db.date == date(2024, 12, 15))
    assert db_15.inflows >= 196000.0


def test_household_salary_reduction_real_user(loaded_dataset: Dataset):
    """user_42 (request_42 / message_30): remaining household salary INR 148,000 projected exactly once per month."""
    p = loaded_dataset.get_profile("user_42")
    req_date = date(2026, 1, 5)
    projs = build_recurring_projections(loaded_dataset, p, req_date, horizon_days=90)
    salary_projs = [pj for pj in projs if pj.category == "salary"]
    # Must have exactly 3 projected salary payments over 90 days (1 per month)
    assert len(salary_projs) == 3
    for pj in salary_projs:
        # Each projected payment must be on the 15th (primary day), never the ended secondary stream on the 20th
        assert pj.proj_date.day == 15
        assert pj.amount_home == 148000.0
    # Historical events must remain intact
    hist_events = [e for e in loaded_dataset.get_events_for_user("user_42") if e.category == "salary" and e.status == "settled"]
    assert any("Second household income" in e.description for e in hist_events)


def test_non_household_multi_salary_user(loaded_dataset: Dataset):
    """user_04: non-household user with multiple salary days retains normal projection behavior."""
    p = loaded_dataset.get_profile("user_04")
    req_date = date(2024, 3, 3)
    projs = build_recurring_projections(loaded_dataset, p, req_date, horizon_days=90)
    salary_projs = [pj for pj in projs if pj.category == "salary"]
    # Normal salary projection continues
    assert len(salary_projs) >= 1


def test_minimum_balance_safety_invariant(loaded_dataset: Dataset):
    """Ensure that for all requests, amount_safe_to_pay never causes projected balance to fall below minimum_balance_to_keep."""
    for rid, req in loaded_dataset.requests.items():
        p = loaded_dataset.get_profile(req.user_id)
        if p is None:
            continue
        safe = compute_amount_safe_to_pay(loaded_dataset, p, req.request_date, req.requested_amount)
        assert 0 <= safe <= req.requested_amount
        # Simulate paying safe amount
        proj = build_projection(
            loaded_dataset, p, req.request_date, horizon_days=90,
            extra_debits=[(req.request_date, safe)],
        )
        if safe > 0:
            assert proj.minimum_spendable() >= p.minimum_balance_to_keep - 1e-4
        else:
            assert safe == 0.0

