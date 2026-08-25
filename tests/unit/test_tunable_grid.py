"""Every slider must be able to represent its own default.

An HTML range input snaps to a grid of `min + n*step`. If the shipped default
is not on that grid the browser silently moves the thumb to the nearest value
that is -- so the page shows 205 while the platform holds 200, the "default"
marker sits somewhere the thumb cannot reach, and double-clicking to reset
appears not to work.

That is not a cosmetic problem: the number on screen is not the number in force,
on the page whose entire job is to tell you what the limits are.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from kat.config.settings import AppSettings
from kat.risk.tunables import TUNABLES, Tunable


@pytest.fixture(scope="module")
def defaults() -> AppSettings:
    return AppSettings()


@pytest.mark.parametrize("tunable", TUNABLES, ids=lambda t: t.key)
class TestSliderGrid:
    def test_the_default_sits_on_the_step_grid(self, tunable: Tunable, defaults):
        value = tunable.read(defaults)
        offset = (value - tunable.minimum) % tunable.step
        assert offset == 0, (
            f"{tunable.key}: default {value} is {offset} off the grid "
            f"(min {tunable.minimum}, step {tunable.step}). The browser would "
            f"snap the thumb to a different number than the platform holds."
        )

    def test_the_default_is_inside_the_bounds(self, tunable: Tunable, defaults):
        value = tunable.read(defaults)
        assert tunable.minimum <= value <= tunable.maximum, (
            f"{tunable.key}: default {value} is outside [{tunable.minimum}, "
            f"{tunable.maximum}] and the slider could not show it"
        )

    def test_the_bounds_are_ordered_and_the_step_fits(self, tunable: Tunable):
        assert tunable.minimum < tunable.maximum, tunable.key
        span = tunable.maximum - tunable.minimum
        assert tunable.step > 0, tunable.key
        # A slider with three positions is a set of radio buttons wearing a
        # costume; one with ten thousand cannot be aimed.
        steps = span / tunable.step
        assert 10 <= steps <= 5000, f"{tunable.key}: {steps} steps is unusable"

    def test_the_step_is_the_one_its_unit_uses(self, tunable: Tunable):
        """Uniform per unit, so two sliders dragged the same distance behave alike."""
        from kat.risk.tunables import _STEP_BY_KIND

        if tunable.step_override is not None:
            return
        assert tunable.step == _STEP_BY_KIND[tunable.kind], tunable.key

    def test_it_has_a_plain_language_explainer(self, tunable: Tunable):
        assert tunable.plain, f"{tunable.key} has no beginner explainer"
        assert len(tunable.plain) > 80, f"{tunable.key}: explainer is too thin to help"


def test_percent_sliders_all_share_one_step():
    """The complaint that started this: percentages felt clicky and uneven."""
    steps = {t.step for t in TUNABLES if t.kind == "percent"}
    assert steps == {Decimal("0.001")}, f"percent steps are not uniform: {steps}"
