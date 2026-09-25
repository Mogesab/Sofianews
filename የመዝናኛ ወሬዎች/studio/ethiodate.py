"""Gregorian -> Ethiopian calendar date, written in Amharic (e.g. "ረቡዕ · መስከረም 13, 2019 ዓ.ም.")."""
from __future__ import annotations

from datetime import date

MONTHS = ["መስከረም", "ጥቅምት", "ኅዳር", "ታኅሣሥ", "ጥር", "የካቲት", "መጋቢት", "ሚያዝያ", "ግንቦት", "ሰኔ", "ሐምሌ", "ነሐሴ", "ጳጉሜ"]
WEEKDAYS = ["ሰኞ", "ማክሰኞ", "ረቡዕ", "ሐሙስ", "ዓርብ", "ቅዳሜ", "እሁድ"]      # Monday first, like date.weekday()
_EPOCH = 1724221                                                          # JDN of 1 Meskerem, year 1 (Amete Mihret)


def to_ethiopian(d: date) -> tuple[int, int, int]:
    jdn = d.toordinal() + 1721425
    r = (jdn - _EPOCH) % 1461
    n = r % 365 + 365 * (r // 1460)
    year = 4 * ((jdn - _EPOCH) // 1461) + r // 365 - r // 1460 + 1
    return year, n // 30 + 1, n % 30 + 1


def label(d: date) -> str:
    y, m, day = to_ethiopian(d)
    return f"{WEEKDAYS[d.weekday()]} · {MONTHS[m - 1]} {day}, {y} ዓ.ም."
