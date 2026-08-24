# Machine Learning

## What the models are for

Estimating fair probabilities, so the platform can compare its number against a
market price and size on the difference. That framing has a sharp consequence:
**calibration matters more than accuracy.**

A model with 65% accuracy that is perfectly calibrated is tradeable — when it says
70%, it happens 70% of the time, so the gap to the market price is real edge. A
model with 80% accuracy that systematically says 90% when it means 70% will lose
money on exactly the trades it is most confident about, which is precisely where
it sizes largest.

Gradient-boosted trees are badly calibrated out of the box — they push
probabilities toward the extremes — so calibrating them is not optional here.

---

## Competing families

No family is assumed best. Four train on identical data and are scored on an
identical held-out split:

| Family | Why it is in the set |
| --- | --- |
| Logistic regression | Well calibrated by construction; a strong baseline that is hard to beat on small, noisy data |
| Random forest | Captures interactions without much tuning; resists overfitting through averaging |
| XGBoost | Usually the strongest ranker; needs firm regularisation on data this small |
| LightGBM | Similar, different splitting strategy; sometimes wins where XGBoost does not |

Which one wins genuinely varies by market and feature set. A platform that
hard-codes "gradient boosting is best" will be wrong on the markets where it is
not.

---

## Splitting is always temporal

`temporal_split()` never shuffles. A random split on time-series data lets a model
learn from Tuesday to predict Monday, which produces a spectacular validation
score and a strategy that loses money.

```
|--------- train 60% ---------|--- validation 20% ---|--- test 20% ---|
                              ^                      ^
                        train ends here        validation ends here
```

The train block always ends before the validation block begins. If timestamps are
supplied the rows are sorted first, so a caller cannot accidentally pass
unordered data.

---

## Information-time integrity

The usual way a research pipeline lies to itself is not a mislabelled column — it
is a feature computed from data the strategy could not actually have held when it
traded.

A one-minute bar closing at 12:00:00 is not knowable at 12:00:00. It is knowable
once the bar has closed, *and* the exchange has published it, *and* our ingestion
has stored it. Backtests that ignore that gap produce Sharpe ratios that evaporate
in paper trading, and the failure is silent because nothing in the data looks
wrong.

Every feature therefore carries:

- an **information timestamp** — the newest observation that went into it
- an **availability lag** — how long after that it could actually be computed

The simulator enforces both: a strategy handed a feature timestamped after the
current simulation time raises rather than trading. Predictions store their
`feature_ts` so leakage is detectable after the fact — a prediction whose feature
timestamp is later than its own timestamp saw the future.

---

## Calibration, and the noise floor

Three numbers are measured on held-out data:

| Metric | What it says |
| --- | --- |
| **Brier** | Mean squared error of probability against outcome. A proper scoring rule: minimised only by reporting your true belief. 0.25 is what always predicting 0.5 gets you |
| **ECE** | Average gap between predicted probability and observed frequency. The number that says "your 70% bucket came in at 55%" |
| **MCE** | The worst single bin. A small ECE can hide a broken tail, and tails are where prediction markets pay |

### Why a fixed ECE threshold is meaningless

A **perfectly calibrated** model shows a non-zero ECE purely from binomial
sampling noise. Measured directly:

| Samples | ECE of a perfectly calibrated model |
| --- | --- |
| 200 | ~0.074 |
| 600 | ~0.042 |
| 1,200 | ~0.029 |
| 5,000 | ~0.015 |
| 20,000 | ~0.007 |

A flat 0.05 gate therefore **rejects perfect calibration** on any realistic
validation set and **accepts sloppy calibration** on a huge one. It is a test of
sample size, not of the model.

So the platform computes the ECE a perfectly calibrated model would show at that
sample size and bin population, and gates on the **excess** over it. That is the
part that would still be there with infinite data — and it is the part with
economic meaning: a systematic two-point error manufactures a two-cent edge on a
$0.50 market and biases every trade the same way, whereas noise averages out
across trades.

### Calibrators

| Method | When it is chosen |
| --- | --- |
| None | The model was already well calibrated; do not make it worse |
| Platt (logistic) | Smooth over- or under-confidence; needs little data |
| Isotonic | Arbitrary monotone shapes; **refused below 1,000 samples** because it fits a step function to noise |

Calibration is fitted on the **validation** split and evaluated on the **test**
split. Fitting and evaluating on the same data would make every model look
perfectly calibrated.

---

## Selection

**Out-of-sample performance only.** Training performance is recorded and used to
*disqualify*, never to win.

### Hard gates

| Gate | Default | Rationale |
| --- | --- | --- |
| Test AUC | ≥ 0.53 | A model that cannot rank cannot be fixed by calibration |
| Excess ECE | ≤ 0.02 | Above this the reported edge is the model's own bias |
| Train − test AUC gap | ≤ 0.15 | Above this the model memorised; its test score is a lucky draw |
| Test rows | ≥ 200 | Below this none of the above means anything |
| Brier vs base rate | Must beat it | Predicting the unconditional frequency requires no model |

A candidate failing any gate is excluded outright rather than compensating
elsewhere. **Returning no model is a legitimate and common outcome** — deploying
the least bad of several failing models is how a platform ends up trading on
noise.

### Observed behaviour

On the validation dataset, LightGBM had the **highest training AUC (0.909)** and
was **rejected** for a 0.152 train-to-test gap. XGBoost was rejected for excess
calibration error 0.032 against a 0.027 noise floor. Logistic regression was
selected on test AUC 0.780. That is selection working: the best fit to the
training data lost.

---

## Registry

Everything needed to reproduce a prediction:

```
model → version → model type → dataset id → feature set id →
train/validation/test windows → metrics per split → calibration →
deployment state → artifact path → SHA-256 → git commit
```

The hash is not ceremony. An artefact whose hash does not match its registry row
has been replaced by something, and `load()` refuses it — a model that changed
after validation is not the model whose metrics justified deploying it.

The calibrator travels with the estimator. A model deployed without the
calibrator fitted for it reports different probabilities from the ones it was
selected on.

### Deployment and rollback

Exactly one model per `(market type, target)` is deployed. Deploying a new one
demotes the incumbent to `archived` rather than deleting it, which makes rollback
a single state change instead of a retraining job. A rolled-back model is marked
`rejected`, so a further rollback does not restore it again.

Event and perpetual models occupy independent slots.

---

## Traceability

Every live inference writes a `Prediction` row: model id and version, feature set
id, feature timestamp, the inputs, the output, and inference latency. Any
prediction can be walked back to exactly what produced it.

---

## Training schedule

Retraining is scheduled, not continuous. `training_interval_s` defaults to six
hours. Continuously retraining on a desktop that is also trading makes the trading
path slow for no measurable benefit.
