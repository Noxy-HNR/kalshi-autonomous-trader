"""Kalshi Autonomous Trader (kat).

A unified platform that trades two independent Kalshi domains -- event contracts
and perpetual futures -- over shared research, data and infrastructure layers.

The hard architectural invariant of this package is defined in
``kat.risk.invariants``: perpetual exposure may never exceed 1.00x account
equity. Nothing in this package is permitted to relax it.
"""

__version__ = "0.1.0"
