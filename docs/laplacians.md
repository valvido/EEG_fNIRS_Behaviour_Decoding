# Graph Laplacian reference (StructuredSparse*)

The **`unmixing.laplacian` config block** controls how electrode/optode *geometry* is turned
into the graph Laplacians `Lx` (EEG) and `Ly` (fNIRS) used by `StructuredSparseCCA` and
`StructuredSparseTCCA`. cedalion shifts `Cxx <- Cxx + alpha_x * Lx` (and `Cyy`/`Ly`)
before whitening, so the smoothness penalty `w^T L w = sum_ij W_ij (w_i - w_j)^2` pulls
the filter weights of *neighbouring* channels toward each other. The block therefore
defines what "neighbouring" means, per modality.

> The `laplacian` block only affects the StructuredSparse* methods. Other methods in the
> same run ignore it, and the Laplacians are only built when a StructuredSparse* method is
> in `methods`. Every builder returns a **symmetric PSD** matrix so `C + alpha*L` stays
> valid for cedalion's whitening.

Source code: [`src/unmixing/laplacian.py`](../src/unmixing/laplacian.py).
Validation: `check_config` in [`src/unmixing/models.py`](../src/unmixing/models.py).
Config keys: [config_reference.md](config_reference.md#laplacian).

---

## Building blocks (graph to Laplacian)

The Laplacians are built once per subject in `unmix`, in the channel order of the
preprocessed data (after pruning), from the first recording. Every method follows the
same two steps:

1. **Adjacency** `W`: a symmetric, non-negative, zero-diagonal weight matrix saying which
   channels are neighbours (and how strongly). This is where the methods differ.
2. **Laplacian** `L`: either the symmetric-normalized `L = I - D^-1/2 W D^-1/2`
   (`normed: true`, the default, spectrum in `[0, 2]`, mean eigenvalue 1) or the
   combinatorial `L = D - W` (`normed: false`).

Use `normed: true`. The preprocessing z-scores every channel, so `Cxx` has mean
eigenvalue about 1, and cedalion adds `l2_reg * L` to it. With `D - W` the scale of `L`
grows with the node degree (on one synthetic subject the mean eigenvalue ranges from 3 for the EEG
eps-ball to 26 for the fNIRS eps-auto graph), so the same `l2_reg` would mean a very
different penalty strength per graph and comparing graphs would mostly compare penalty
strengths.

Channel positions (mm):

- **EEG**: the electrode positions set by the subject's `montage.eeg`
  (`eeg_geo3d` of the recording).
- **fNIRS**: source-detector **midpoints** from the subject's optode positions
  (`geo3d` of the recording).

### Distance metric (`distance`)

Position-based methods accept:

| Value       | What it measures |
|-------------|------------------|
| `euclidean` | Straight-line 3D distance (default). |
| `geodesic`  | Great-circle distance on a least-squares sphere fitted to the layout. Avoids Euclidean "tunnelling through the skull" between opposite-side channels. |

---

## Methods

Registry: `GRAPH_BUILDERS`. `delaunay` needs an electrode layout (meant for EEG),
`shared_optode` needs the source and detector of each channel (fNIRS only).

| Method                | EEG | fNIRS | Weights | Knobs |
|-----------------------|:---:|:-----:|---------|-------|
| `euclidean_eps`       | yes | yes   | binary  | `eps`, `distance` |
| `gaussian_normalized` | yes | yes   | Gaussian | `sigma`, `eps` *or* `k`, `distance` |
| `knn`                 | yes | yes   | binary  | `k`, `distance` |
| `delaunay`            | yes | no    | binary  | *(none)* |
| `shared_optode`       | no  | yes   | binary  | *(none)* |

### `euclidean_eps`: eps-ball graph (the default)

Channels closer than `eps` (mm) are neighbours (`W_ij = 1` iff `D_ij < eps`).

- `eps`: positive number in mm, or `"auto"` = the maximum edge of the minimum spanning
  tree, i.e. the **smallest radius that keeps the graph connected**, derived from the
  data instead of hand-picked.

### `gaussian_normalized`: heat-kernel weights

Smooth weights `W_ij = exp(-D_ij^2 / 2 sigma^2)`: close channels are coupled strongly,
distant ones weakly, with no hard cutoff unless you add one.

- `sigma`: bandwidth in mm, or `"median"` (default) = the median of the positive pairwise
  distances.
- Optional sparsification (at most one applies; `k` wins over `eps`):
  - `k`: keep only each channel's k nearest neighbours (symmetrized).
  - `eps`: zero out edges with `D >= eps` (`"auto"` as above).

### `knn`: k-nearest-neighbour graph

Each channel connects to its `k` nearest neighbours; the adjacency is symmetrized
(`i` and `j` are connected if *either* is in the other's k-NN set). Adapts to local
channel density where a global radius cannot. Default `k = 6`; capped at `N - 1`.

### `delaunay`: triangulation topology (EEG)

Parameter-free: positions are projected onto their two principal axes (a cap is roughly
a 2D surface) and Delaunay-triangulated; triangle edges are the graph edges. Gives a
natural planar "mesh" of the cap without choosing any radius or k. Fewer than 3 channels
fall back to a fully connected graph.

### `shared_optode`: measurement topology (fNIRS only)

Two fNIRS channels are neighbours iff they share a **source or a detector**, as the
data's channel optodes (`fnirs_optodes`, set by the montage) say; the channel names
play no part. This is the natural measurement topology: such channels
sample overlapping tissue. Parameter-free; ignores positions entirely.

---

## Defaults

With an empty `laplacian` block, or for the modality a block leaves out:

```yaml
eeg:   { method: euclidean_eps, distance: euclidean, normed: true, eps: 60 }
fnirs: { method: euclidean_eps, distance: euclidean, normed: true, eps: 25 }
```

This is the original notebook graph (binary eps-ball, Euclidean, eps in mm), but
normalized; the notebook used the unnormalized `D - W`. A block given for a modality
replaces that modality's default entirely, so give every key you need.

---

## Using it in a config

The block sits in the `unmixing` section of the experiment config. Per modality, the keys
are the arguments of `build_laplacian`: `method`, `distance`, `normed`, `eps`, `k`,
`sigma`. Before any data is loaded, `check_config` rejects blocks other than `eeg` and
`fnirs` and unknown `method` names. Other mistakes (an unknown key, `shared_optode` for
EEG) only fail when the Laplacians are built.

```yaml
unmixing:
  # Topology-based (parameter-free)
  laplacian:
    eeg:   { method: delaunay, normed: true }
    fnirs: { method: shared_optode, normed: true }

  # Data-driven eps-ball on geodesic distances
  # laplacian:
  #   eeg:   { method: euclidean_eps, distance: geodesic, eps: auto, normed: true }
  #   fnirs: { method: euclidean_eps, distance: geodesic, eps: auto, normed: true }

  # Gaussian weights, sparsified by k-NN
  # laplacian:
  #   eeg:   { method: gaussian_normalized, k: 6, sigma: median, normed: true }
  #   fnirs: { method: gaussian_normalized, k: 4, sigma: median, normed: true }
```

Tip: list the available methods with
`python -c "from src.unmixing.laplacian import list_builders; print(list_builders())"`.
