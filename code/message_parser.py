"""
message_parser.py -- Deterministic message parsing and financial context extraction.

Extracts structured financial facts from messages.csv:
1. Salary Increase: Permanent monthly raise from effective date onward.
2. Temporary / Reduced Salary: Reduced salary baseline confirmation.
3. Employment / Seasonal Contract Termination: Stops recurring salary extrapolation.
4. Household Income Reduction: Sets remaining household monthly salary baseline.
5. Rent / Lease Increase: 12% multiplier on forward recurring rent debits.
6. Salary Payment Date Shift: Specific replacement date for upcoming salary.
7. Confirmed Client Invoice: One-time confirmed cash inflow on settlement date.
8. Pending Credits: Excluded / ignored per challenge rules.
9. Fraud / Scam: Excluded / ignored completely.
10. Bilingual Support: English and Indonesian templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

from data_loader import Dataset, FinancialProfile, Message


@dataclass
class MessageFinancialContext:
    """Structured financial facts extracted from a user's messages."""
    user_id: str
    salary_increase: Optional[Tuple[float, str, date]] = None   # (amount, currency, effective_date)
    temporary_salary: Optional[Tuple[float, str]] = None         # (amount, currency)
    household_salary: Optional[Tuple[float, str]] = None         # (amount, currency)
    salary_ended: bool = False                                    # contract/job ended
    rent_multiplier: float = 1.0                                  # e.g. 1.12
    salary_date_shift: Optional[date] = None                     # replacement date for upcoming salary
    confirmed_invoices: List[Tuple[float, str, date]] = field(default_factory=list) # [(amount, currency, settlement_date)]


def _parse_num(s: str) -> float:
    """Parse numeric amount from regex match group, stripping trailing punctuation."""
    return float(s.rstrip('.'))


def parse_message_text(text: str, user_id: str, ctx: MessageFinancialContext) -> None:
    """
    Parse a single message text and populate the MessageFinancialContext.
    Uses deterministic keyword and regex matching for English and Indonesian templates.
    """
    t = text.lower()

    # 1. Scam / Fraud messages -- ignore completely (never create income/expenses)
    if any(k in t for k in [
        'release charge', 'biaya pencairan',
        'processing fee to claim', 'biaya administrasi untuk klaim'
    ]):
        return

    # 2. Salary Increase
    # EN: "Your monthly salary has increased to USD 2988. The change applies from 2026-07-15."
    # ID: "Gaji bulanan Anda naik menjadi IDR 42750000. Perubahan ini berlaku mulai 2025-08-15."
    m_inc = re.search(
        r'(?:monthly salary has increased to|gaji bulanan anda (?:telah )?naik menjadi)\s+([a-z]{3})\s+([\d\.]+).*?(?:applies from|mulai)\s+(\d{4}-\d{2}-\d{2})',
        text, re.IGNORECASE | re.DOTALL
    )
    if m_inc:
        amt = _parse_num(m_inc.group(2))
        cur = m_inc.group(1).upper()
        eff_date = date.fromisoformat(m_inc.group(3))
        ctx.salary_increase = (amt, cur, eff_date)
        return

    # 3. Household Income Reduction
    # EN: "One household employment record has ended. The remaining confirmed monthly salary is INR 148000."
    # ID: "Salah satu sumber pendapatan kerja rumah tangga telah berakhir. Sisa gaji bulanan yang dikonfirmasi adalah IDR 25840000."
    m_hh = re.search(
        r'(?:remaining confirmed monthly salary is|sisa gaji bulanan yang dikonfirmasi adalah)\s+([a-z]{3})\s+([\d\.]+)',
        text, re.IGNORECASE
    )
    if m_hh:
        amt = _parse_num(m_hh.group(2))
        cur = m_hh.group(1).upper()
        ctx.household_salary = (amt, cur)
        return

    # 4. Temporary / Reduced Salary
    # EN: "Your temporary monthly pay is EUR 1037.52." / "Your next salary is reduced to USD 702."
    # ID: "Gaji bulanan sementara Anda adalah IDR 31464000." / "Gaji berikutnya dikurangi menjadi..."
    m_temp = re.search(
        r'(?:temporary monthly pay is|gaji bulanan sementara anda adalah|next salary is reduced to|gaji berikutnya dikurangi menjadi)\s+([a-z]{3})\s+([\d\.]+)',
        text, re.IGNORECASE
    )
    if m_temp:
        amt = _parse_num(m_temp.group(2))
        cur = m_temp.group(1).upper()
        ctx.temporary_salary = (amt, cur)
        return

    # 5. Employment / Seasonal Contract Termination
    # EN: "The current seasonal contract has ended.", "Your employment has ended."
    # ID: "Kontrak musiman saat ini telah berakhir.", "Hubungan kerja telah berakhir."
    if any(k in t for k in [
        'seasonal contract has ended', 'kontrak musiman saat ini telah berakhir',
        'employment has ended', 'hubungan kerja telah berakhir'
    ]):
        ctx.salary_ended = True
        return

    # 6. Rent / Lease Increase (+12%)
    # EN: "The renewed lease increases monthly rent by 12%."
    # ID: "Perpanjangan sewa menaikkan biaya sewa bulanan sebesar 12%."
    if any(k in t for k in [
        'increases monthly rent by 12%',
        'menaikkan biaya sewa bulanan sebesar 12%'
    ]):
        ctx.rent_multiplier = 1.12
        return

    # 7. Salary Payment Date Shift
    # EN: "Your confirmed salary is now expected on 2024-09-23. This replaces the payroll date shown in the earlier update."
    # ID: "Gaji yang telah dikonfirmasi sekarang diperkirakan cair pada 2024-09-23."
    m_shift = re.search(
        r'(?:salary is now expected on|gaji yang telah dikonfirmasi sekarang diperkirakan cair pada)\s+(\d{4}-\d{2}-\d{2})',
        text, re.IGNORECASE
    )
    if m_shift:
        ctx.salary_date_shift = date.fromisoformat(m_shift.group(1))
        return

    # 8. Confirmed Client Invoice
    # EN: "The client approved an invoice payment of INR 196000. Settlement is expected on 2024-12-15; the other submitted invoices are still awaiting approval."
    # ID: "Klien menyetujui pembayaran faktur sebesar IDR 15390000. Penyelesaian diperkirakan pada 2025-08-15;"
    m_inv = re.search(
        r'(?:approved an invoice payment of|menyetujui pembayaran faktur sebesar)\s+([a-z]{3})\s+([\d\.]+).*?(?:settlement is expected on|penyelesaian diperkirakan pada)\s+(\d{4}-\d{2}-\d{2})',
        text, re.IGNORECASE | re.DOTALL
    )
    if m_inv:
        amt = _parse_num(m_inv.group(2))
        cur = m_inv.group(1).upper()
        settle_date = date.fromisoformat(m_inv.group(3))
        ctx.confirmed_invoices.append((amt, cur, settle_date))
        return


def get_message_context_for_user(dataset: Dataset, user_id: str) -> MessageFinancialContext:
    """
    Retrieve and parse all messages for a user into a structured MessageFinancialContext.
    Messages are processed chronologically by sent_at.
    """
    ctx = MessageFinancialContext(user_id=user_id)
    messages = dataset.get_messages_for_user(user_id)
    if not messages:
        return ctx

    sorted_msgs = sorted(messages, key=lambda m: m.sent_at)
    for m in sorted_msgs:
        parse_message_text(m.message_text, user_id, ctx)
    return ctx
