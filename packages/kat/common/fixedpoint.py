"""Exact fixed-point arithmetic for prices, contract counts and money.

Why this module exists
----------------------
Every risk decision in this platform -- above all the 1.00x perpetual leverage
invariant -- is an *exact* comparison between two monetary quantities. Binary
floating point cannot represent 0.01 exactly, so a naive ``float`` pipeline can
compute an effective leverage of 1.0000000000000002 for a position that is in
truth exactly 1.0x, or (far worse) 0.9999999999999999 for one that is truly over
the line. We therefore never use ``float`` for money.

Representation
--------------
Three integer scales are used throughout the codebase. They are chosen so that
the product of a price and a quantity is itself exact and needs no rounding:

``price_cc``  -- 1e-4 USD, "centicents". Kalshi's finest documented tick is
                 $0.0001 (the ``center_centi_edge_centi_cent`` structure), so
                 every valid price on every price-level structure is an exact
                 integer number of centicents.
``qty_qc``    -- 1e-2 contracts, "centi-contracts". Kalshi fixed-point contract
                 counts carry at most 2 decimals and the documented minimum
                 granularity is 0.01 contracts.
``micros``    -- 1e-6 USD. Monetary amounts. Kalshi emits dollar values with up
                 to 6 decimals, and price x quantity lands exactly here because
                 1e-4 * 1e-2 == 1e-6.

Consequently ``notional_micros = price_cc * qty_qc`` is an exact integer
multiplication, and the leverage invariant reduces to the integer comparison
``exposure_micros <= equity_micros`` -- no division, no float, no epsilon.

Parsing external data
---------------------
Kalshi sends money as decimal *strings* (``"0.5600"``). We parse them with
``decimal.Decimal`` (exact for decimal literals) and then quantise. When a value
carries more precision than our scale can hold we never round to nearest
silently: the caller chooses a direction, so that risk inputs can always be
rounded *against* the trader (equity down, exposure up).
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import Final

__all__ = [
    "CC_PER_DOLLAR",
    "MICROS_PER_DOLLAR",
    "QC_PER_CONTRACT",
    "FixedPointError",
    "Rounding",
    "cc_to_micros",
    "dollars_to_micros",
    "format_count",
    "format_dollars",
    "micros_to_decimal",
    "notional_micros",
    "parse_count_qc",
    "parse_dollars_micros",
    "parse_price_cc",
    "price_cc_to_decimal",
    "qc_to_decimal",
]

# --- scale constants -------------------------------------------------------

CC_PER_DOLLAR: Final[int] = 10_000         # price scale:    $1.00 == 10000 cc
QC_PER_CONTRACT: Final[int] = 100          # quantity scale: 1 contract == 100 qc
MICROS_PER_DOLLAR: Final[int] = 1_000_000  # money scale:    $1.00 == 1000000 micros

_D_CC: Final[Decimal] = Decimal(CC_PER_DOLLAR)
_D_QC: Final[Decimal] = Decimal(QC_PER_CONTRACT)
_D_MICROS: Final[Decimal] = Decimal(MICROS_PER_DOLLAR)

_Q_DOLLARS_4: Final[Decimal] = Decimal("0.0001")
_Q_DOLLARS_6: Final[Decimal] = Decimal("0.000001")
_Q_COUNT_2: Final[Decimal] = Decimal("0.01")


class Rounding:
    """Direction to take when an external value is finer than our scale.

    ``FLOOR`` and ``CEIL`` exist so that safety-critical parsing can always be
    biased against the trader. ``EXACT`` raises instead of losing information and
    is the right choice when a silent adjustment would mask a schema change.
    """

    FLOOR: Final[str] = "floor"
    CEIL: Final[str] = "ceil"
    HALF_EVEN: Final[str] = "half_even"
    EXACT: Final[str] = "exact"


class FixedPointError(ValueError):
    """Raised when a value cannot be represented at the requested scale."""


def _to_decimal(value: str | int | float | Decimal, *, field: str) -> Decimal:
    """Coerce an external value to ``Decimal``, rejecting anything non-finite.

    ``float`` is accepted only as a convenience for tests and synthetic data; it
    is converted through ``str`` so that ``0.07`` means the literal 0.07 rather
    than its binary expansion.
    """
    if isinstance(value, Decimal):
        d = value
    elif isinstance(value, bool):  # bool is an int subclass; never a valid amount
        raise FixedPointError(f"{field}: boolean is not a numeric amount")
    elif isinstance(value, int):
        d = Decimal(value)
    elif isinstance(value, float):
        d = Decimal(str(value))
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            raise FixedPointError(f"{field}: empty string is not a numeric amount")
        try:
            d = Decimal(s)
        except InvalidOperation as exc:
            raise FixedPointError(f"{field}: {value!r} is not a decimal number") from exc
    else:
        raise FixedPointError(f"{field}: unsupported type {type(value).__name__}")

    if not d.is_finite():
        raise FixedPointError(f"{field}: {value!r} is not finite")
    return d


def _quantise_to_int(d: Decimal, scale: Decimal, rounding: str, field: str) -> int:
    """Scale ``d`` by ``scale`` and convert to ``int`` under ``rounding``."""
    scaled = d * scale
    if rounding == Rounding.EXACT:
        if scaled != scaled.to_integral_value():
            raise FixedPointError(
                f"{field}: {d} carries more precision than the 1/{scale} scale allows"
            )
        return int(scaled)
    mode = {
        Rounding.FLOOR: ROUND_FLOOR,
        Rounding.CEIL: ROUND_CEILING,
        Rounding.HALF_EVEN: ROUND_HALF_EVEN,
    }.get(rounding)
    if mode is None:
        raise FixedPointError(f"{field}: unknown rounding mode {rounding!r}")
    return int(scaled.to_integral_value(rounding=mode))


# --- parsing ---------------------------------------------------------------


def parse_price_cc(
    value: str | int | float | Decimal,
    *,
    rounding: str = Rounding.EXACT,
    field: str = "price",
) -> int:
    """Parse a Kalshi ``FixedPointDollars`` price into centicents.

    Defaults to ``EXACT``: a price that is not on the 1/10000 grid indicates
    either a new price-level structure or a corrupted payload, and both deserve
    a loud failure rather than a quiet rounding.
    """
    return _quantise_to_int(_to_decimal(value, field=field), _D_CC, rounding, field)


def parse_count_qc(
    value: str | int | float | Decimal,
    *,
    rounding: str = Rounding.EXACT,
    field: str = "count",
) -> int:
    """Parse a Kalshi ``FixedPointCount`` into centi-contracts (signed)."""
    return _quantise_to_int(_to_decimal(value, field=field), _D_QC, rounding, field)


def parse_dollars_micros(
    value: str | int | float | Decimal,
    *,
    rounding: str = Rounding.HALF_EVEN,
    field: str = "amount",
) -> int:
    """Parse a Kalshi ``FixedPointDollars`` money amount into micro-dollars.

    Risk-critical callers should pass ``Rounding.FLOOR`` for anything that
    increases capacity (equity, balances) and ``Rounding.CEIL`` for anything that
    consumes it (notional, margin, fees).
    """
    return _quantise_to_int(_to_decimal(value, field=field), _D_MICROS, rounding, field)


# Friendlier alias used by config parsing and tests.
dollars_to_micros = parse_dollars_micros


# --- conversions -----------------------------------------------------------


def notional_micros(price_cc: int, qty_qc: int) -> int:
    """Exact notional in micro-dollars for ``qty_qc`` contracts at ``price_cc``.

    1e-4 USD * 1e-2 contracts == 1e-6 USD, so this is a plain integer product
    with no rounding step at all. ``qty_qc`` may be signed; the result carries
    the same sign, and risk callers take ``abs()`` themselves so that the sign
    convention stays explicit at the call site.
    """
    return price_cc * qty_qc


def cc_to_micros(price_cc: int) -> int:
    """Convert a centicent price into micro-dollars (per single contract)."""
    return price_cc * 100


def price_cc_to_decimal(price_cc: int) -> Decimal:
    """Centicents -> exact ``Decimal`` dollars, for display and reporting."""
    return (Decimal(price_cc) / _D_CC).quantize(_Q_DOLLARS_4)


def qc_to_decimal(qty_qc: int) -> Decimal:
    """Centi-contracts -> exact ``Decimal`` contract count."""
    return (Decimal(qty_qc) / _D_QC).quantize(_Q_COUNT_2)


def micros_to_decimal(micros: int) -> Decimal:
    """Micro-dollars -> exact ``Decimal`` dollars."""
    return (Decimal(micros) / _D_MICROS).quantize(_Q_DOLLARS_6)


# --- formatting for the wire ----------------------------------------------


def format_dollars(price_cc: int) -> str:
    """Render a centicent price as the 4-decimal string Kalshi expects."""
    return f"{price_cc_to_decimal(price_cc):.4f}"


def format_count(qty_qc: int) -> str:
    """Render a centi-contract count as the 2-decimal string Kalshi expects."""
    return f"{qc_to_decimal(qty_qc):.2f}"
