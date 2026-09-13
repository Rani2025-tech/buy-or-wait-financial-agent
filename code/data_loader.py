"""
data_loader.py — Load and index all dataset CSV files.

Provides O(1) lookups by request_id, user_id, event_id, etc.
Does NOT modify any file inside dataset/.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_REPO_ROOT, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")


def _csv_path(name: str) -> str:
    return os.path.join(DATASET_DIR, name)


# ---------------------------------------------------------------------------
# Raw row types (plain dicts from csv.DictReader, typed via dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: List[str]
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[int]  # None means user won't consider installments


@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str          # "credit" | "debit"
    amount: Optional[float] # None = blank, must be resolved from image
    currency: str
    event_date: date
    settlement_date: Optional[date]
    status: str             # settled|pending|scheduled|cancelled|failed|unrealized
    linked_event_id: str    # may be empty string
    flexibility: str        # fixed|reducible|stoppable|reducible_or_stoppable|""
    minimum_allowed_amount: Optional[float]


@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str          # full_payment | installments
    payment_amount: float
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: Optional[int]  # blank for full_payment (single payment)
    financing_fee: float
    total_payable_amount: float


@dataclass
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float


@dataclass
class Message:
    message_id: str
    user_id: str
    request_id: str         # may be empty
    related_event_id: str   # may be empty
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass
class ImageRecord:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str   # links to FinancialEvent.event_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> Optional[date]:
    s = s.strip()
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def _parse_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    return float(s)


def _parse_int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    return int(s)


def _parse_datetime(s: str) -> datetime:
    """Parse ISO 8601 datetime strings in either 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SSZ' format."""
    s = s.strip().replace("T", " ").rstrip("Z")
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _pipe_list(s: str) -> List[str]:
    """Split a pipe-separated string into a list, filtering empty strings."""
    return [x.strip() for x in s.split("|") if x.strip()]


# ---------------------------------------------------------------------------
# Dataset — the single object that holds everything
# ---------------------------------------------------------------------------

class Dataset:
    """
    Holds all loaded data and provides indexed lookups.
    Load once at startup; reuse throughout the program.
    """

    def __init__(self) -> None:
        # Primary stores
        self.requests: Dict[str, Request] = {}
        self.profiles: Dict[str, FinancialProfile] = {}
        self.events: Dict[str, FinancialEvent] = {}          # event_id → event
        self.payment_options: Dict[str, PaymentOption] = {}  # option_id → option
        self.messages: Dict[str, Message] = {}               # message_id → message
        self.images: Dict[str, ImageRecord] = {}             # image_id → record
        self.exchange_rates: List[ExchangeRate] = []

        # Secondary indexes
        self._events_by_user: Dict[str, List[FinancialEvent]] = defaultdict(list)
        self._options_by_request: Dict[str, List[PaymentOption]] = defaultdict(list)
        self._messages_by_user: Dict[str, List[Message]] = defaultdict(list)
        self._messages_by_request: Dict[str, List[Message]] = defaultdict(list)
        self._messages_by_event: Dict[str, List[Message]] = defaultdict(list)
        self._images_by_event: Dict[str, ImageRecord] = {}   # event_id → image
        self._images_by_request: Dict[str, List[ImageRecord]] = defaultdict(list)
        # FX index: (from_currency, to_currency) → sorted list of (rate_date, rate)
        self._fx_index: Dict[tuple, List[tuple]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        self._load_requests()
        self._load_profiles()
        self._load_events()
        self._load_payment_options()
        self._load_exchange_rates()
        self._load_messages()
        self._load_images()

    def _load_requests(self) -> None:
        with open(_csv_path("requests.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = Request(
                    request_id=row["request_id"],
                    user_id=row["user_id"],
                    request_date=_parse_date(row["request_date"]),
                    request_type=row["request_type"],
                    requested_amount=float(row["requested_amount"]),
                    desired_completion_date=_parse_date(row["desired_completion_date"]),
                    allows_partial_payment=row["allows_partial_payment"].strip().lower() == "true",
                    request_text=row["request_text"],
                )
                self.requests[r.request_id] = r

    def _load_profiles(self) -> None:
        with open(_csv_path("financial_profiles.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = FinancialProfile(
                    user_id=row["user_id"],
                    home_currency=row["home_currency"],
                    current_available_balance=float(row["current_available_balance"]),
                    minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                    financial_priorities=_pipe_list(row["financial_priorities"]),
                    expense_categories_to_protect=_pipe_list(row["expense_categories_to_protect"]),
                    expense_categories_user_is_willing_to_reduce=_pipe_list(
                        row["expense_categories_user_is_willing_to_reduce"]
                    ),
                    expense_categories_user_is_willing_to_stop=_pipe_list(
                        row["expense_categories_user_is_willing_to_stop"]
                    ),
                    payment_methods_user_will_consider=_pipe_list(
                        row["payment_methods_user_will_consider"]
                    ),
                    max_installment_months=_parse_int(row["max_installment_months"]),
                )
                self.profiles[p.user_id] = p

    def _load_events(self) -> None:
        with open(_csv_path("financial_events.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                e = FinancialEvent(
                    event_id=row["event_id"],
                    user_id=row["user_id"],
                    event_type=row["event_type"],
                    description=row["description"],
                    category=row["category"],
                    direction=row["direction"],
                    amount=_parse_float(row["amount"]),
                    currency=row["currency"],
                    event_date=_parse_date(row["event_date"]),
                    settlement_date=_parse_date(row["settlement_date"]),
                    status=row["status"],
                    linked_event_id=row["linked_event_id"].strip(),
                    flexibility=row["flexibility"].strip(),
                    minimum_allowed_amount=_parse_float(row["minimum_allowed_amount"]),
                )
                self.events[e.event_id] = e
                self._events_by_user[e.user_id].append(e)

    def _load_payment_options(self) -> None:
        with open(_csv_path("request_payment_options.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                o = PaymentOption(
                    payment_option_id=row["payment_option_id"],
                    request_id=row["request_id"],
                    payment_method=row["payment_method"],
                    payment_amount=float(row["payment_amount"]),
                    number_of_payments=int(row["number_of_payments"]),
                    first_payment_date=_parse_date(row["first_payment_date"]),
                    payment_frequency_days=_parse_int(row["payment_frequency_days"]),
                    financing_fee=float(row["financing_fee"]),
                    total_payable_amount=float(row["total_payable_amount"]),
                )
                self.payment_options[o.payment_option_id] = o
                self._options_by_request[o.request_id].append(o)

    def _load_exchange_rates(self) -> None:
        with open(_csv_path("exchange_rates.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                er = ExchangeRate(
                    rate_date=_parse_date(row["rate_date"]),
                    from_currency=row["from_currency"],
                    to_currency=row["to_currency"],
                    rate=float(row["rate"]),
                )
                self.exchange_rates.append(er)
                key = (er.from_currency, er.to_currency)
                self._fx_index[key].append((er.rate_date, er.rate))
        # Sort each list by date ascending
        for key in self._fx_index:
            self._fx_index[key].sort(key=lambda x: x[0])

    def _load_messages(self) -> None:
        with open(_csv_path("messages.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = Message(
                    message_id=row["message_id"],
                    user_id=row["user_id"],
                    request_id=row["request_id"].strip(),
                    related_event_id=row["related_event_id"].strip(),
                    sent_at=_parse_datetime(row["sent_at"].strip()),
                    source_type=row["source_type"],
                    message_text=row["message_text"],
                )
                self.messages[m.message_id] = m
                self._messages_by_user[m.user_id].append(m)
                if m.request_id:
                    self._messages_by_request[m.request_id].append(m)
                if m.related_event_id:
                    self._messages_by_event[m.related_event_id].append(m)

    def _load_images(self) -> None:
        with open(_csv_path("images.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img = ImageRecord(
                    image_id=row["image_id"],
                    user_id=row["user_id"],
                    request_id=row["request_id"].strip(),
                    related_event_id=row["related_event_id"].strip(),
                )
                self.images[img.image_id] = img
                if img.related_event_id:
                    self._images_by_event[img.related_event_id] = img
                if img.request_id:
                    self._images_by_request[img.request_id].append(img)

    # ------------------------------------------------------------------
    # Public lookup API
    # ------------------------------------------------------------------

    def get_request(self, request_id: str) -> Optional[Request]:
        return self.requests.get(request_id)

    def get_profile(self, user_id: str) -> Optional[FinancialProfile]:
        return self.profiles.get(user_id)

    def get_event(self, event_id: str) -> Optional[FinancialEvent]:
        return self.events.get(event_id)

    def get_events_for_user(self, user_id: str) -> List[FinancialEvent]:
        return self._events_by_user.get(user_id, [])

    def get_payment_options_for_request(self, request_id: str) -> List[PaymentOption]:
        return self._options_by_request.get(request_id, [])

    def get_messages_for_user(self, user_id: str) -> List[Message]:
        return self._messages_by_user.get(user_id, [])

    def get_messages_for_request(self, request_id: str) -> List[Message]:
        return self._messages_by_request.get(request_id, [])

    def get_messages_for_event(self, event_id: str) -> List[Message]:
        return self._messages_by_event.get(event_id, [])

    def get_image_for_event(self, event_id: str) -> Optional[ImageRecord]:
        return self._images_by_event.get(event_id)

    def get_images_for_request(self, request_id: str) -> List[ImageRecord]:
        return self._images_by_request.get(request_id, [])

    def get_fx_series(self, from_currency: str, to_currency: str) -> List[tuple]:
        """Return sorted list of (rate_date, rate) for a currency pair."""
        return self._fx_index.get((from_currency, to_currency), [])

    def image_path(self, image_id: str) -> str:
        """Return the filesystem path for a given image_id."""
        return os.path.join(IMAGES_DIR, f"{image_id}.png")
