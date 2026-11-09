# Config reference

Every setting the pipeline reads, file by file. The settings are defined, with their
checks, in [`src/config/schema.py`](../src/config/schema.py); this page explains them.
A run saves every resolved value to `experiment.yaml` and `subject.yaml` in its
results folder, so you can always see which values a past run used.

## How the files fit together

One run takes two arguments: an experiment and a subject.

```
python main.py --config configs/experiments/<experiment>.yaml \
               --subject configs/subjects/<format>/<subject_id>/<dataset>.yaml
```

| File | What it says | How often it changes |
|---|---|---|
| `configs/experiments/*.yaml` | What to run: hand (the hand chosing is not fully implemented, its a part of the old code and should be revisited before running with 
just one hand), unmixing, classification. Points to the next three files. | Every experiment |
| `configs/pruning/*.yaml` | Bad-channel detection. | Rarely |
| `configs/preprocessing/*.yaml` | Signal preprocessing. Two variants: `full_recording.yaml` (CCA family) and `mspoc.yaml`. | Rarely |
| `configs/unmixing/grids.yaml` | Hyperparameter grids of every method. | Rarely |
| `configs/subjects/<format>/<subject_id>/*.yaml` | Which data: subject, paths, groups, test group. | One file per dataset |

Rules that hold for all files:

- Every key lives in exactly one file. A key in the wrong file is an unknown key there,
  and loading stops with the file and key named.
- A missing key takes its default (the "Default" column below). A key without a default
  must be given.
- Lists are written `[a, b]`; a pair such as `[lo, hi]` must have `lo < hi`.
- Paths to the pruning, preprocessing and grids files are relative to the experiment
  file. Data paths in a subject file are relative to the project root.

---

## Experiment file (`configs/experiments/*.yaml`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | text | required | Experiment name. Part of the results path: `results/runs/<name>/<dataset>/<start time>/`, which holds `results/` (the data) and `figures/` (every standard figure, always drawn), where `<dataset>` is the subject file's `name`. Every saved run is also listed in `results/runs/index.csv`. |
| `hand` | `left`, `right`, `both` | `both` | Which hand(s) the experiment is about. `both` means two ground-truth sources (right_hand and left_hand), the others one. |
| `pruning` | path | required | The pruning file. |
| `preprocessing` | path | required | The preprocessing file. |
| `unmixing` | section | required | See [unmixing](#unmixing-section). |
| `classification` | section | absent | See [classification](#classification-section). Absent means no classification. |
| `save_inputs` | true/false | `true` | Also save the preprocessed runs every model was fit on, to `data/preprocessed/<dataset>/<start time>/` (not to the results; `run.json` points there). Stored as float32: about 7 MB per band-power run of a synthetic subject, about 50 MB with mSPoC's band-passed EEG. Turn it off once the inputs are checked, e.g. for SNR sweeps. The run's figures never need this: they are drawn from the runs in memory. Without saved inputs, `load_inputs` has nothing to read, so the input plots of notebook 05 cannot be drawn later. |

### `unmixing` section

Non-nested evaluation always runs: the subject's `test_group` is held out, each method's
hyperparameters are selected by leave-one-group-out cross-validation on the other
groups, and the selected model is refit on them and scored once on the test group.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `grids` | path | required | The grids file, see [grids](#grids-file-configsunmixinggridsyaml). |
| `methods` | list | required | Methods to run, each once: `CCA`, `SparseCCA`, `RidgeCCA`, `ElasticNetCCA`, `StructuredSparseCCA`, `tCCA`, `ElasticNetTCCA`, `StructuredSparseTCCA`, `mSPoC`. mSPoC needs the mSPoC preprocessing file (`eeg_feature: bandpassed`); all others need `eeg_feature: bandpower`. |
| `metrics` | list | `[joint_avg]` | Selection metrics. Each picks one hyperparameter combination per method, so each gives its own selected model. See [metrics](#selection-metrics). |
| `nested` | true/false | `false` | Also run nested evaluation: the same selection with every group held out once, for analysis only. It never produces the model classification uses. |
| `n_components` | number or `null` | `null` | Canonical pairs to fit. `null` fits all. Must be at least `assignment.n_top`. mSPoC needs a number (cedalion's mSPoC cannot fit all). |
| `assignment` | section | see below | How the unlabelled pairs are assigned to the hands. |
| `grid_overrides` | mapping | `{}` | Replace whole keys of a method's grid for this experiment, in the grids file format, e.g. `{tCCA: {time_shifts: [[0, 5]]}}`. Only for methods in `methods`. |
| `laplacian` | mapping | `{}` | Graph Laplacians of the StructuredSparse methods: `{eeg: {...}, fnirs: {...}}`. Empty uses the default graph. See [Laplacian](#laplacian). |
| `seed` | integer | `0` | Base seed. Every fit derives its own seed from it, the same for any `n_jobs`. |
| `n_jobs` | integer | `1` | Fits in parallel. `-1` uses all cores. Results do not depend on it. |

#### `assignment`

A fitted model gives unlabelled pairs, ordered by canonical correlation; the EEG and
fNIRS component of one pair always go together. On the training data, a strategy
decides which pair is the right hand and which the left hand
([`src/unmixing/assignment.py`](../src/unmixing/assignment.py)). The same pairs are then
scored on the test data.

Strategies (`method`):

- `ground_truth` (synthetic data only): among the first `n_top` pairs, the assignment
  with the highest score

      sum over hands of  eeg_weight * |r(EEG component, EEG source)|
                       + fnirs_weight * |r(fNIRS component, fNIRS source)|

- `motor_channels` (no ground truth needed): the same search, with the score

      sum over hands of  eeg_weight * share_eeg + fnirs_weight * share_fnirs

  where share is the mean over the hand's motor channels of `|a_c| / ||a||`, and `a` is
  the column of the estimated mixing (Haufe pattern) of the pair's component on the
  training runs. The channels come from a motor channels file, one per montage
  ([`configs/motor_channels/`](../configs/motor_channels/)).
- `task_regressors` (no ground truth needed): like `ground_truth`, with each hand's
  task references in place of its sources. A hand's trials are the trial types of its
  side in the subject's `trial_classes`. The EEG reference is a boxcar over each trial,
  the fNIRS reference the boxcar convolved with the gamma HRF `fnirs_hrf` (cedalion's
  `hrf_regressors`). Every trial counts with amplitude 1, so only the cue timing goes in.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `method` | string | `ground_truth` | The strategy. |
| `n_top` | integer | `2` | The hands are assigned among the first `n_top` pairs. With two sources and `n_top: 2` this only decides which of the two pairs is the right hand and which the left hand. Must be at least the number of sources (2 for `hand: both`, else 1). |
| `eeg_weight` | number | `1.0` | Weight of the EEG term in the score, for every strategy. |
| `fnirs_weight` | number | `1.0` | Weight of the fNIRS term. Both 0 is not allowed. With `motor_channels`, a modality with weight 0 is not used. |
| `motor_channels` | path | none | `motor_channels` only: the motor channels file, relative to the experiment file. It maps modality (`eeg`, `fnirs`) to hand to channel names. Channels missing from the data (e.g. pruned) are skipped. |
| `fnirs_hrf` | mapping | `{tau: 0.0, sigma: 3.0, T: 0.0}` | `task_regressors` only: the gamma HRF of the fNIRS reference in seconds, the arguments of cedalion's `glm.Gamma`. |

Independently of `n_top`, every run also reports, per source, which pair out of all
fitted pairs matched it best (`best_pair_<source>` in `scores.csv`). It is a diagnostic
only and never used for selection.

#### Selection metrics

Each metric turns the scores of one validation fold into one number (higher is
better); the combination with the best mean over the folds is selected. Each quantity
of a metric is averaged over the folds on its own (correlations through Fisher's z), then
the quantities are averaged plainly; see [metrics.md](metrics.md).

| Name | Value | Ground truth |
|---|---|---|
| `joint_avg` | Mean of \|r\| between each labelled pair's EEG and fNIRS component and its source, over sources and both modalities. | Needed |
| `isi_avg` | 1 minus the mean Amari ISI of the labelled pairs' filters against the ground-truth mixing, over both modalities. 1 is perfect separation. Not defined for one source. | Needed |
| `canonical_correlation_avg` | Mean of the signed correlation between the EEG and fNIRS component of each labelled pair; a pair whose modalities move against each other counts against it. | Not used in the value, but labelling still needs it |

#### Laplacian

Only the StructuredSparse methods use it. Each modality's block is the arguments of
one graph builder; see [laplacians.md](laplacians.md) for what each builder does.

| Key | Values | Meaning |
|---|---|---|
| `method` | `euclidean_eps`, `gaussian_normalized`, `knn`, `delaunay` (EEG), `shared_optode` (fNIRS) | How neighbours are found. |
| `distance` | `euclidean`, `geodesic` | Distance between channel positions, for the position-based builders. |
| `normed` | true/false | Normalised Laplacian instead of degree minus adjacency. |
| `eps` | number (mm) or `auto` | Neighbour radius (`euclidean_eps`, optional sparsifier of `gaussian_normalized`). |
| `k` | integer | Neighbours per channel (`knn`, optional sparsifier of `gaussian_normalized`). |
| `sigma` | number or `median` | Width of the Gaussian weights (`gaussian_normalized`). |

Default when `laplacian` is empty: `euclidean_eps`, euclidean, not normalised, with
`eps` 60 mm (EEG) and 25 mm (fNIRS). EEG positions are the electrode positions set
by the subject's `montage.eeg`, fNIRS positions the source-detector midpoints of the
optode positions set by `montage.fnirs`.

### `classification` section

Present means: after unmixing, classify the trials from the recovered sources with
shrinkage LDA (rLDA). `classification: {}` uses all defaults. Every selected model
unmixes each run on its own; each trial is cut into an epoch from the earliest to
the latest time any window below needs, its baseline mean is subtracted, and the
mean of each interval of each labelled pair's component is one feature. The
classifier is fit on the trials of the groups the model was fit on and scored on
the trials of the held-out group: the test group, and with `nested: true` every
group once. An epoch that reaches past the end (or start) of its run is kept: the
baseline and every interval are averaged over the part the run has. Only a trial
whose baseline or a whole interval has no data (e.g. in the NaN end of a temporal
model's EEG component) is dropped, and counted. A class without training or test trials stops the run. If
the epochs reach past the shortest gap between two onsets, the run logs a warning
and goes on.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `classes` | list of trial types | `[left, right]` | The classes, at least two, e.g. `[left, right, rest]`. |
| `rest_class` | text | `rest` | The trial type that means rest. A run whose stim table has such trials uses them; otherwise rest epochs are cut from the stretches no stimulus's epoch covers, as many as fit, back to back. |
| `feature_sets` | list of `eeg_fnirs`, `eeg`, `fnirs` | all three | One classifier per feature set, all on the same trials. |
| `baseline_window` | `[lo, hi]` s | `[-2.0, 0.0]` | Relative to the stimulus onset; its mean is subtracted from each epoch. |
| `eeg_intervals` | list of `[t0, t1]` s | `[[0.5, 4.5], [5.5, 8.0]]` | After onset; the mean of each interval is one EEG feature. |
| `fnirs_intervals` | list of `[t0, t1]` s | `[[2.0, 5.0], [5.0, 8.0], [8.0, 12.0]]` | The same for fNIRS. |

---

## Grids file (`configs/unmixing/grids.yaml`)

One block per method. Every combination of a method's values is tried in the
selection. An experiment can replace whole keys with `unmixing.grid_overrides`.

| Key | Format | Meaning |
|---|---|---|
| `l1_reg` | `{eeg: [...], fnirs: [...]}` | Sparsity penalty per modality. Every (EEG, fNIRS) pair of values is tried, e.g. 5 x 5 = 25 combinations. |
| `l2_reg` | `{eeg: [...], fnirs: [...]}` | Ridge penalty per modality, crossed the same way. |
| `time_shifts` | list of shift sets, in seconds | Temporal methods only (tCCA family and mSPoC): each set is one candidate time embedding of the EEG, e.g. `[[0, 7], [0, 5], [0, 3, 5, 7]]`. 0 is added by cedalion if missing. |
| `max_iter`, `N_restarts`, `tol` | list of values | mSPoC only: iterations per restart, random restarts, convergence tolerance. |

Each method accepts only its own keys; any other key is rejected before any data is read:

| Method | Keys |
|---|---|
| `CCA` | none |
| `SparseCCA` | `l1_reg` |
| `RidgeCCA` | `l2_reg` |
| `ElasticNetCCA`, `StructuredSparseCCA` | `l1_reg`, `l2_reg` |
| `tCCA` | `time_shifts` (required) |
| `ElasticNetTCCA`, `StructuredSparseTCCA` | `l1_reg`, `l2_reg`, `time_shifts` (required) |
| `mSPoC` | `time_shifts` (required), `max_iter`, `N_restarts`, `tol` |

Fixed settings are never grid keys: `shift_source` (always false; the pipeline shifts the
EEG components with cedalion's `shift_by_optimal` where it needs them on the EEG's time), `scale` (always true),
`N_components` (`unmixing.n_components`), the Laplacians (`unmixing.laplacian`) and the
seed (`unmixing.seed`).

A method without hyperparameters (`CCA`) has an empty block: `CCA: {}`. The number of
fits per method is (combinations) x (validation folds), plus one refit per selected
combination: ElasticNetCCA with the current grid is 25 x 16 = 400 combinations.

---

## Pruning file (`configs/pruning/*.yaml`)

Finds the subject's bad channels once and removes them from every run: bad EEG channels
are interpolated, bad fNIRS channels dropped.

The EEG is judged whole, on each recording on its own: for synthetic data on the background
recording of each group, whose runs all get its list, otherwise on the recording itself. A
channel bad in a recording is interpolated in that recording only. A bad motor electrode
(`eeg.motor_channels`) in any recording stops the subject. The fNIRS is judged on a window of
the rest period of each recording the reader selects, 3 s left out at each end of that
period: the first minute of the earliest background recording (synthetic), the minute of
rest before the first instruction of every session (`shin2017`), or the whole session
(`buccino`, which has no rest period long enough). A fNIRS channel bad in any of them is
dropped from every run.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `random_state` | integer | `42` | Seed for pyprep's RANSAC. |
| `window_sec` | number (s) | `60.0` | Seconds of rest the fNIRS is judged on, from the rest onset. Never runs past the end of rest. |
| `eeg` | section | see below | EEG detection. |
| `fnirs` | section | see below | fNIRS detection. |

### `eeg`

Two pyprep passes with an average re-reference in between. A channel flagged in either
pass is bad.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `line_freqs` | list (Hz) | `[50, 100]` | Notched before detection. |
| `ransac` | true/false | `true` | Run the RANSAC criterion in pass 2. |
| `motor_channels` | list | `[C3, C4]` | Electrodes over the hand motor areas. Any of them bad in any recording stops the subject; each must be in the EEG montage (`shin2017`: FCC3h, CCP3h, FCC4h, CCP4h). |
| `pass1.flat_threshold` | number (V) | `1.0e-15` | SD or MAD below this counts as flat. |
| `pass1.deviation_threshold` | number | `5.0` | Robust z-score of the channel amplitude. |
| `pass1.hf_noise_threshold` | number | `5.0` | Robust z-score of the noise-to-signal ratio above 50 Hz. |
| `pass2.correlation_secs` | number (s) | `1.0` | Window length of the correlation criterion. |
| `pass2.correlation_threshold` | number | `0.4` | Lowest acceptable correlation with the best-matching other channel. |
| `pass2.correlation_frac_bad` | number | `0.1` | Fraction of bad windows that makes a channel bad (also for dropout). |
| `pass2.ransac_n_samples` | integer | `50` | RANSAC: number of random channel subsets. |
| `pass2.ransac_sample_prop` | number | `0.25` | RANSAC: fraction of channels per subset. |
| `pass2.ransac_corr_thresh` | number | `0.75` | RANSAC: lowest acceptable correlation with the prediction. |
| `pass2.ransac_frac_bad` | number | `0.4` | RANSAC: fraction of bad windows that makes a channel bad. |
| `pass2.ransac_window_secs` | number (s) | `5.0` | RANSAC: window length. |

### `fnirs`

Judged on a cleaned amplitude: the whole recording is converted to optical
density, motion corrected (TDDR, then wavelet) and high-pass filtered, then cut to the
window and converted back to amplitude. A channel is kept only if it passes every
check.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `highpass_hz` | number (Hz) | `0.01` | High-pass cutoff in optical density. |
| `butter_order` | integer | `4` | Order of that Butterworth filter. |
| `sd_range_mm` | `[lo, hi]` mm | `[10, 45]` | Allowed source-detector distance. Drops the short channels (about 8 mm). |
| `amp_range_v` | `[lo, hi]` V | `[0.001, 0.85]` | Allowed mean amplitude. |
| `snr_threshold` | number | `3.0` | Lowest mean / standard deviation of the amplitude. |
| `sci_threshold` | number | `0.5` | Lowest scalp coupling index, per window. |
| `psp_threshold` | number | `0.05` | Lowest peak spectral power, per window. |
| `window_length_s` | number (s) | `5.0` | Window for SCI and PSP. |
| `time_pass_fraction` | number | `0.5` | Fraction of windows in which a channel must pass SCI and PSP together. |

---

## Preprocessing file (`configs/preprocessing/*.yaml`)

Turns each pruned run into the signals the models are fit on. The models always use
the whole run, cut to the stimulus window; the input is never cut into trials.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `eeg_feature` | `bandpower`, `bandpassed` | `bandpower` | `bandpower` (CCA family): power envelope of `eeg.feature_band` (the squared Hilbert envelope), on the fNIRS time points. `bandpassed` (mSPoC): the band-filtered EEG at its own rate. |
| `fnirs_srate` | number (Hz) | `7.4` | Rate the EEG band power is brought to before it is put on the fNIRS time points. |
| `downsample_method` | `resample`, `epoch_mean` | `resample` | How the band power is brought to `fnirs_srate`. |
| `trim_pre_sec` | number (s) | `5.0` | Kept from the first stimulus onset minus this. |
| `trim_post_sec` | number (s) | `20.0` | Kept until the last stimulus end plus this. `full_recording.yaml` sets 8.0, `mspoc.yaml` 20.0. |
| `fnirs_mspoc_srate` | number (Hz) or `null` | `null` | mSPoC only: fNIRS and ground truth are downsampled to this rate, which sets mSPoC's block length (one EEG block per fNIRS sample). Must divide the EEG rate into whole blocks. `mspoc.yaml` sets 1.0. |
| `eeg` | section | see below | EEG cleaning and feature. |
| `fnirs` | section | see below | fNIRS conversion and filter. |
| `checks` | section | see below | When a run counts as badly cut or aligned. |

### `eeg`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `bandpass` | `[lo, hi]` Hz | `[1.0, 45.0]` | Broadband filter before ICA. |
| `ica_n_components` | integer | `15` | ICA components to fit; capped at the number of channels that were not interpolated. |
| `ica_method` | text | `picard` | ICA algorithm, as MNE names it (`picard`, `fastica`, `infomax`). |
| `random_state` | integer | `42` | Seed for ICA. |
| `feature_band` | `[lo, hi]` Hz | `[8.0, 12.0]` | Band of the EEG feature (alpha). The EEG ground truth is taken from the same band. |
| `drop_interpolated` | true/false | `false` | Drop the interpolated channels from the EEG feature. `mspoc.yaml` sets true: interpolated channels make the covariance of band-passed EEG singular, which mSPoC cannot fit on. |

With `eeg_feature: bandpassed` a run also stops if ICA finds eye components to remove,
for the same reason (removing them makes the covariance singular).

### `fnirs`

| Key | Type | Default | Meaning |
|---|---|---|---|
| `dpf` | number | `6.0` | Differential pathlength factor, both wavelengths. |
| `bandpass` | `[lo, hi]` Hz | `[0.01, 0.6]` | Band kept of the HbO signal. The fNIRS ground truth is filtered the same way. |

### `checks`

A run that fails stops the pipeline.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `min_window_coverage` | number | `0.9` | Fraction of the stimulus window the kept data must cover. |

---

## Subject file (`configs/subjects/<format>/<subject_id>/*.yaml`)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `subject_id` | text | required | Subject ID; used to find the data files. |
| `name` | text | the file's name | Name of this dataset, e.g. `sub01_snr_fnirs-5_eeg5`. Part of the results path, so runs at different SNRs get their own folders. |
| `format` | text | required | Which reader loads the data: `synthetic`, `buccino` (Buccino et al. 2016) or `shin2017` (Shin et al. 2017, dataset A). |
| `paths` | mapping | required | Data locations; the keys depend on the format. `synthetic` needs `data` (the simulator's output folder) and `background` (the folder of the subject's background recordings). `buccino` needs `mat` (the subject's MAT file). `shin2017` needs `eeg` and `nirs`, the subject's two folders of BBCI files (`cnt.mat`, `mrk.mat`, and `mnt.mat` for the NIRS). |
| `montage` | mapping | required | Electrode and optode positions; see below. The only place they are chosen. |
| `groups` | list | required | Groups (background recordings) whose runs are used, each once. They are the cross-validation folds. For a session recorded in one piece, its blocks: the session is preprocessed whole and then split into one run per listed block, in recording order. Buccino: `block_1`, `block_2`, ... Shin: `session<k>_block<b>`, two blocks of 10 consecutive trials (5 per hand) in each motor imagery session k = 1, 3, 5, e.g. `session3_block2`. |
| `test_group` | text or `null` | `null` | The group held out in non-nested evaluation. Must be one of `groups`; unmixing needs it. |

### `montage`

Loading applies it once; the positions then travel with the data (electrodes in the
EEG, optodes as `geo3d`), are saved with the run in `geometry.nc` and are what pruning
interpolates with, the Laplacians are built from and the head maps are drawn on.

| Key | Type | Meaning |
|---|---|---|
| `eeg.standard` | text | An MNE template montage by name, e.g. `standard_1020`. |
| `eeg.file` | path | A montage file that comes with the dataset, in a format `mne.channels.read_custom_montage` reads (.elc, .sfp, .bvef, ...). |
| `fnirs.recording` | `true` | The optode positions stored in the recording's own files (the SNIRF). |
| `fnirs.file` | path | An optode position file, as `cedalion.io.load_tsv` reads it (e.g. `data/montage/*.tsv`). |
| `fnirs.layout` | path | Which optodes form each channel, as template sites: a tab-separated file with the columns `channel`, `source`, `detector` (e.g. `C3A  FCC5h  FCC3h`); lines starting with `#` are comments. Needs `fnirs.standard`. |
| `fnirs.standard` | text | With `fnirs.layout`: the MNE template montage that places the sites, e.g. `standard_1005`. |

`eeg` sets exactly one of `standard` and `file`; `fnirs` exactly one of `recording`,
`file` and `layout`. EEG channel names are spelled as in the montage after loading
(e.g. `Fc5` becomes `FC5`); a channel the montage does not place stops loading.
Without `fnirs` the subject cannot be loaded. With `fnirs.layout`, every fNIRS channel
gets the layout's source and detector, so channels share optodes as on the real probe,
and the sites are placed in the same head frame as the EEG electrodes.

```yaml
montage:                  # synthetic data
  eeg:
    standard: standard_1020
  fnirs:
    recording: true

montage:                  # Buccino: optodes reconstructed from Fig 1 of the paper
  eeg:
    standard: standard_1005
  fnirs:
    layout: configs/montages/buccino_fnirs.tsv
    standard: standard_1005

montage:                  # Shin: optode sites of each channel from the dataset's mnt.mat
  eeg:
    standard: standard_1005
  fnirs:
    layout: configs/montages/shin2017_fnirs.tsv
    standard: standard_1005
```

Runs saved before the montage was part of the subject have no EEG positions; loading
them warns and uses `standard_1020`, which they all used.

Not written in the file: for synthetic data, the simulator's own config
(`{paths.data}/simulation_config.yaml`) is read and saved with the run as `simulation`.

---

## Fixed values that are not settings

Values the code uses that deliberately have no setting, with the reason:

| Value | Where | Why fixed |
|---|---|---|
| fNIRS mixing for ISI: HbO at the first wavelength | `src/unmixing/scores.py` | The two wavelengths differ by one constant factor (1.805 on one synthetic subject), which the ISI does not depend on. |
| At least 11 shared samples for a correlation | `src/unmixing/scores.py` | Below that a correlation is not meaningful; it is NaN instead. |
| mSPoC block tolerance: 5% of a block | `src/unmixing/fit.py` | The recorded EEG rates are not exact (e.g. 250.0032 Hz for 250 Hz); blocks drift a few ms per run. The error it guards against was a whole block. |
| Plots orient each component so its pattern is positive at its largest channel | `src/visualization/orient.py` | Display only; scores and saved results keep the model's sign. Open question for the supervisor. |
