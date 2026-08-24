# Contributing

This is a private, single-operator trading platform. These notes are for future
me as much as anyone.

## Before changing anything

Read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for setup, conventions and how to
run the tests.

## The one rule

**Never weaken the 1.00x perpetual leverage invariant.**

`packages/kat/risk/invariants.py` contains no leverage parameter by design. If a
change to that file is what makes your work pass, stop and reconsider what you
are making pass.

Changes that make it *stricter* are welcome. Changes that add a configuration
knob, a caller-supplied maximum, a bypass for an "emergency", or a special case
for a particular strategy are not, however convenient.

## Checklist for a change

- [ ] `pytest` passes
- [ ] `pytest tests/leverage` passes
- [ ] `ruff check packages/ apps/ tests/ migrations/` is clean
- [ ] New behaviour has a test that would fail without it
- [ ] A strategy change has tests for both firing *and* declining
- [ ] `alembic check` is clean if models changed
- [ ] No secret is in the diff (`git check-ignore -v .env`)

## Commit messages

Say what changed and why it matters. If a bug was fixed, say what the bug
actually did -- "the strategy never fired at any funding rate" is useful; "fix
funding bug" is not.

## What not to add

- Docker, containers, or a shell script the platform needs at runtime
- A paid service as a mandatory dependency
- A float anywhere on the money path
- A wall-clock read in strategy or simulation code
- Any path from a strategy, model, LLM or the dashboard to the exchange that does
  not pass through a risk engine
