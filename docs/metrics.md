# Selection metrics reference

A **selection metric** decides which hyperparameter combination of a method is selected.
Every combination is scored in every validation fold, the metric turns each fold's scores
into one number (higher is better), the numbers are averaged over the folds, and the
combination with the best mean wins. You set them with `unmixing.metrics` in the
experiment config (default `[joint_avg]`); each metric gives its own selected model.

> Selection metrics are not the reported scores. A metric only *chooses* a combination.
> Every selected model reports **all** scores (`scores.csv`: \|r\| per source and
> modality, ISI, canonical correlation, best pairs), whichever metric selected it.

Source code: [`src/unmixing/metrics.py`](../src/unmixing/metrics.py) (metrics),
[`src/unmixing/scores.py`](../src/unmixing/scores.py) (the scores they read),
[`src/unmixing/select.py`](../src/unmixing/select.py) (the selection loop).
Config keys: [config_reference.md](config_reference.md#selection-metrics).

---

## How one fold is scored

Validation folds are leave-one-group-out over the dev groups (every group except the
subject's `test_group`). In each fold, and the same way for the final model on the test
group:

1. The model is fit on the training runs, joined on one time axis. The components keep the
   sign the model gives them; no score depends on it (see [Signs](#signs)).
2. **Assignment, on the training runs:** the strategy `assignment.method` gives each hand
   (right_hand, left_hand) its own pair ([`src/unmixing/assignment.py`](../src/unmixing/assignment.py)).
   With `ground_truth`, the components of every pair are correlated with the
   ground-truth sources and the assignment among the first `assignment.n_top` pairs with
   the highest `sum over hands of eeg_weight * |r_eeg| + fnirs_weight * |r_fnirs|` wins.
   Without ground truth, `motor_channels` scores the pairs by where their estimated
   mixing lies, and `task_regressors` by their correlation with each hand's trials
   (see [config_reference](config_reference.md#assignment)). A pair is never split: a hand gets the EEG and the fNIRS component of one pair.
3. **Scoring, on the held-out runs:** the labelled pairs are applied and compared with the
   ground truth there. This gives a `Scores`:

| Field | What it measures |
|---|---|
| `r_eeg[source]`, `r_fnirs[source]` | \|r\| of the source's EEG (fNIRS) component with the EEG (fNIRS) ground truth. |
| `isi_eeg`, `isi_fnirs` | Amari ISI of the labelled pairs' filters W against the ground-truth mixing A of the features the model is fit on, `G = \|W' A\|`. 0 is perfect separation, 1 none. For EEG band power, A is the square of the EEG field, `A_eeg ** 2`: a source adds band power to a channel in proportion to the square of its field there, whatever the sign of the field. For band-passed EEG (mSPoC), A is `A_eeg`. For fNIRS, A is the HbO mixing at the first wavelength. NaN with one source. |
| `canonical_r[source]` | **Signed** correlation between the source's pair's EEG and fNIRS component: the pair's canonical correlation as the model found it on this split. For the temporal methods the EEG component is taken as the model pairs it, before it is shifted back to the EEG's time: the time-embedded EEG at t against the fNIRS at t. Negative means the two modalities moved against each other here. Uses no ground truth. |
| `best_pairs` | Diagnostic only, never used for selection: per source, the pair out of **all** fitted pairs that matched it best on the training data. |
| `pattern_r_eeg[source]`, `pattern_r_fnirs[source]` | Diagnostic only, never used for selection: \|r\| of the labelled pair's Haufe pattern on the held-out runs (fitted filter, held-out data covariance) with the source's ground-truth mixing, over the channels. The mixing is taken as for the ISI: `A_eeg ** 2` for EEG band power, `A_eeg` for band-passed EEG, HbO at the first wavelength for fNIRS. Computed only for the selected models and their fold refits; in `landscape.csv` it is NaN for every other combination. |
| `leak_eeg[source]`, `leak_fnirs[source]` | Diagnostic only, never used for selection: \|r\| of the EEG (fNIRS) component of the pair assigned to the **other** hand with this hand's ground truth: how much of this hand leaks into the other pair. NaN with one hand. |

Correlations use only the samples where both signals are finite. If a labelled pair has
fewer than 11 shared samples with its source, or between its EEG and fNIRS component, the
fit or the data is broken and the run stops with an error naming the method, setting and
fold. For pairs that are not labelled, too few samples just give NaN.

For the temporal methods (tCCA family and mSPoC), the EEG component exists in two versions.
cedalion fits u(t) = sum over tau of w_tau' x(t - tau) to match the fNIRS component at t, so this
version is on the fNIRS time and is the one compared with the fNIRS component (`canonical_r`).
cedalion's `shift_by_optimal` moves it back by the pair's optimal shift tau_k to the time of the
EEG it came from; that version is compared with the EEG ground truth, plotted and classified, and
its last tau_k has no data and is NaN. Both versions are set to NaN within the largest time shift
after the start of every run, the first included, because there the time embedding reaches into
the previous run or into cedalion's zero padding.

---

## The metrics

Each metric is built from **quantities**, one value per fold each:

| Metric | Quantities | Averaged in | Ground truth |
|---|---|---|---|
| `joint_avg` | \|r_eeg\| and \|r_fnirs\| of each source: four with two sources | Fisher's z | Needed |
| `isi_avg` | `1 - isi_eeg`, `1 - isi_fnirs`, so 1 is perfect separation | plain mean | Needed (mixing A) |
| `canonical_correlation_avg` | signed `canonical_r` of each labelled pair | Fisher's z | Not in the value, but the pairs it uses are labelled with it |

`joint_avg` takes absolute values: a pair with the right time course but the opposite sign
counts as recovered.

`canonical_correlation_avg` keeps the sign: a pair whose EEG and fNIRS move against each
other on held-out data counts against the combination, as in the legacy pipeline.

### Signs

A fitted model gives each component as s or -s with the same fit. Unmixing keeps the sign
the model gives: the input data and the ground truth are never flipped, labelling and
every score use \|r\|, and the saved components and patterns are the model's own. Only
`canonical_r` is signed, since it compares the model's EEG and fNIRS component with each
other, not with the ground truth.

The plots flip each component and its pattern together so the pattern is positive at its
largest channel ([`src/visualization/orient.py`](../src/visualization/orient.py)), only
so the colours can be read. mSPoC's EEG component is band power, which does not change
sign with its filter, so its time course is never flipped; its pattern is.

### Averaging over folds

`fold_mean` gives the value selection compares, in two steps:

1. **Each quantity over the folds,** on its own. For correlations through Fisher's z,
   `tanh(mean(arctanh(values)))` with values clipped just inside (-1, 1): the folds are
   repeated estimates of the same correlation, which is what Fisher's z averages. ISI
   values as a plain mean.
2. **The plain mean of the quantities.** They are different quantities, not estimates of
   one correlation, so they are not Fisher-averaged together.

A combination is only compared when it could be judged on every fold. If a fit failed in
any fold, or a value is missing in a fold (e.g. the ISI with one source), its `fold_mean` is
NaN and it cannot be selected; the log names it and the fold's error is in `landscape.csv`.

A fit fails, among other reasons, when the model's filters are NaN (`DegenerateFit`). Sparse
CCA does this when the L1 soft threshold removes every weight of a filter: cedalion then
normalises a zero vector, and the NaN carries through the deflation into every later pair.
If the chosen combination's final fit on all dev groups is degenerate, that method selects
nothing for that metric and a warning says so.

The value of a metric in a single fold (`landscape.csv`, the dots in the landscape
figures) is the plain mean of its quantities in that fold. It is for display only.

### Selection

For each metric, the combination with the highest `fold_mean` is selected (`np.nanargmax`;
ties go to the first combination in grid order). It is refit on all dev groups and scored
once on the test group. Metrics that pick the same combination share one refit. If no
combination of a method could be scored for a metric, that metric selects nothing for that
method and a warning is logged.

---

## One source (`hand: left` or `right`)

The single source is whichever hand the run contains, read from the ground truth.

- `joint_avg` is the mean of that source's \|r_eeg\| and \|r_fnirs\|.
- `isi_avg` is NaN, because the ISI needs at least two sources. It never selects anything,
  so leave it out of `metrics`.
- `canonical_correlation_avg` uses the one labelled pair.

---

## Using them in a config

```yaml
unmixing:
  metrics: [joint_avg, isi_avg, canonical_correlation_avg]   # two sources
  # metrics: [joint_avg, canonical_correlation_avg]          # one source
```

Unknown names are rejected before any data is loaded. The available names are the keys
of `METRICS` in [`src/unmixing/metrics.py`](../src/unmixing/metrics.py).
