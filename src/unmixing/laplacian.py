"""Configurable graph Laplacians (Lx/Ly) for the structured-sparse CCA models.

`StructuredSparseCCA` / `StructuredSparseTCCA` add a graph-smoothness penalty to the CCA
objective: cedalion shifts ``Cxx <- Cxx + alpha_x * Lx`` (and likewise ``Cyy``/``Ly``)
before whitening, so the penalty ``w^T L w = sum_ij W_ij (w_i - w_j)^2`` pulls the filter
weights of neighbouring channels toward each other. `Lx` indexes EEG channels, `Ly`
indexes fNIRS channels, both in the data's channel order.

This module turns electrode/optode *geometry* into those Laplacians, with the graph
construction selectable per modality from a config block (see `build_laplacians`). Every
builder returns a **symmetric PSD** matrix so that ``Cxx + alpha*L`` stays positive
(semi-)definite for cedalion's `inv_sqrt_cov` whitening.

Methods (registry `GRAPH_BUILDERS`):
  * ``euclidean_eps``       — binary eps-ball graph, ``L = D - A`` (the legacy notebook
                              construction). ``eps`` is a number or ``"auto"`` (data-driven:
                              the MST max-edge, i.e. the smallest threshold that connects
                              the graph).
  * ``gaussian_normalized`` — Gaussian/heat-kernel weights ``exp(-d^2 / 2 sigma^2)``
                              (``sigma="median"`` by default), optionally sparsified by
                              ``eps``/``k``; pair with ``normed: true`` for the symmetric
                              normalized Laplacian.
  * ``knn``                 — symmetric k-nearest-neighbour graph (no distance threshold).
  * ``delaunay``            — parameter-free neighbour graph from a Delaunay triangulation
                              (EEG only).
  * ``shared_optode``       — fNIRS channels sharing a source or detector are neighbours
                              (the natural measurement topology; fNIRS only).

Distance metric (``distance``) for the position-based builders is ``euclidean`` or
``geodesic`` (great-circle distance on a least-squares-fitted sphere, which avoids 3D
Euclidean "tunnelling through the skull").
"""
import numpy as np
import xarray as xr
from scipy.sparse.csgraph import laplacian as _csgraph_laplacian, minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

# Default used for StructuredSparse* when no `laplacian` block is given: binary eps-ball,
# Euclidean, eps in mm (EEG=60, fNIRS=25), normalized so the mean eigenvalue is 1 and
# l2_reg means the same penalty strength for every graph.
DEFAULT_LAPLACIAN = {
    "eeg":   {"method": "euclidean_eps", "distance": "euclidean", "normed": True, "eps": 60},
    "fnirs": {"method": "euclidean_eps", "distance": "euclidean", "normed": True, "eps": 25},
}


# --------------------------------------------------------------------------------------
# Distance + geometry helpers
# --------------------------------------------------------------------------------------
def _fit_sphere(positions):
    """Least-squares sphere fit. Returns (center (3,), radius float).

    Algebraic fit: minimise ||A x - b|| with A=[2X,2Y,2Z,1], b=x^2+y^2+z^2; then
    R = sqrt(f + |c|^2). One consistent radius keeps geodesic distances in mm.
    """
    P = np.asarray(positions, dtype=float)
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    b = (P ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    R = float(np.sqrt(max(sol[3] + c @ c, 1e-12)))
    return c, R


def _pairwise_distance(positions, metric):
    """(N, N) symmetric distance matrix (mm) for a position-based builder."""
    P = np.asarray(positions, dtype=float)
    if metric == "euclidean":
        return cdist(P, P)
    if metric == "geodesic":
        c, R = _fit_sphere(P)
        U = P - c
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U = U / norms
        cos = np.clip(U @ U.T, -1.0, 1.0)
        D = R * np.arccos(cos)
        np.fill_diagonal(D, 0.0)
        return D
    raise ValueError(f"Unknown distance metric {metric!r}; expected 'euclidean' or 'geodesic'.")


def _auto_eps(D):
    """Smallest eps that connects the graph = max edge of the MST (slightly inflated so
    the strict ``D < eps`` comparison includes that edge)."""
    mst = minimum_spanning_tree(D).toarray()
    m = float(mst.max())
    return m * (1.0 + 1e-6) if m > 0 else 1.0


def _knn_mask(D, k):
    """Symmetric binary k-NN adjacency from a distance matrix (self excluded)."""
    N = D.shape[0]
    k = int(min(k, N - 1)) if N > 1 else 0
    A = np.zeros_like(D)
    if k > 0:
        # column 0 of argsort is the node itself (distance 0); take the next k.
        idx = np.argsort(D, axis=1)[:, 1:k + 1]
        rows = np.repeat(np.arange(N), k)
        A[rows, idx.ravel()] = 1.0
        A = np.maximum(A, A.T)
    return A


def _resolve_sigma(D, sigma):
    """Gaussian bandwidth: a number, or the median of positive pairwise distances."""
    if isinstance(sigma, str) and sigma == "median":
        iu = np.triu_indices_from(D, k=1)
        d = D[iu]
        d = d[np.isfinite(d) & (d > 0)]
        return float(max(np.median(d), 1e-6)) if d.size else 1.0
    return float(sigma)


# --------------------------------------------------------------------------------------
# Adjacency/weight builders -> (W, builder_info). All return symmetric, non-negative,
# zero-diagonal weight matrices.
# --------------------------------------------------------------------------------------
def _adj_euclidean_eps(positions, distance="euclidean", eps=None, **_):
    D = _pairwise_distance(positions, distance)
    eps_used = _auto_eps(D) if (eps is None or eps == "auto") else float(eps)
    W = (D < eps_used).astype(float)
    np.fill_diagonal(W, 0.0)
    return W, {"eps_used": float(eps_used)}


def _adj_gaussian(positions, distance="euclidean", eps=None, k=None, sigma="median", **_):
    D = _pairwise_distance(positions, distance)
    sig = _resolve_sigma(D, sigma)
    W = np.exp(-(D ** 2) / (2.0 * sig ** 2))
    np.fill_diagonal(W, 0.0)
    info = {"sigma_used": float(sig)}
    if k is not None:                                   # sparsify by k-NN
        W = W * _knn_mask(D, k)
        info["k_used"] = int(min(k, len(D) - 1))
    elif eps is not None:                               # sparsify by radius
        eps_used = _auto_eps(D) if eps == "auto" else float(eps)
        W = W * (D < eps_used)
        info["eps_used"] = float(eps_used)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W, info


def _adj_knn(positions, distance="euclidean", k=None, **_):
    D = _pairwise_distance(positions, distance)
    k = 6 if k is None else k
    W = _knn_mask(D, k)
    return W, {"k_used": int(min(k, len(D) - 1)) if len(D) > 1 else 0}


def _adj_delaunay(positions, **_):
    """Parameter-free neighbour graph from a 2D Delaunay triangulation. Points are
    projected onto their two principal axes first (an electrode cap is ~a 2D surface)."""
    P = np.asarray(positions, dtype=float)
    N = len(P)
    A = np.zeros((N, N))
    if N < 3:
        # Too few points to triangulate: connect them all.
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)
        return A, {}
    Pc = P - P.mean(axis=0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    P2 = Pc @ Vt[:2].T
    tri = Delaunay(P2)
    for simplex in tri.simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    A[a, b] = 1.0
    return A, {}


def _adj_shared_optode(optodes=None, **_):
    """fNIRS channels sharing a source OR a detector are neighbours. `optodes` holds
    the (source, detector) of every channel, in channel order."""
    if optodes is None:
        raise ValueError("shared_optode requires the optodes of every channel.")
    src = [source for source, _ in optodes]
    det = [detector for _, detector in optodes]
    N = len(optodes)
    A = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and (src[i] == src[j] or det[i] == det[j]):
                A[i, j] = 1.0
    return A, {}


GRAPH_BUILDERS = {
    "euclidean_eps":       _adj_euclidean_eps,
    "gaussian_normalized": _adj_gaussian,
    "knn":                 _adj_knn,
    "delaunay":            _adj_delaunay,
    "shared_optode":       _adj_shared_optode,
}


def list_builders():
    return sorted(GRAPH_BUILDERS)


# --------------------------------------------------------------------------------------
# Laplacian assembly
# --------------------------------------------------------------------------------------
def build_laplacian(positions, *, method, channels=None, optodes=None, distance="euclidean",
                    normed=True, eps=None, k=None, sigma="median"):
    """Build one symmetric-PSD graph Laplacian for a set of channels.

    Args:
        positions: (N, 3) channel coordinates in mm, in channel order.
        method:    key into `GRAPH_BUILDERS`.
        channels:  channel names.
        optodes:   (source, detector) of every channel, in channel order (required
                   for ``shared_optode``).
        distance:  ``"euclidean"`` | ``"geodesic"`` (position-based builders).
        normed:    if True, symmetric-normalized Laplacian ``I - D^-1/2 W D^-1/2``
                   (spectrum in [0, 2], mean eigenvalue 1); else combinatorial
                   ``D - W``, whose scale grows with the node degree. Default True, so
                   that l2_reg means the same penalty strength across graphs.
        eps, k, sigma: method-specific knobs.

    Returns:
        (L, info) where L is (N, N) symmetric PSD and info records the resolved params
        and node/edge counts.
    """
    if method not in GRAPH_BUILDERS:
        raise KeyError(f"Unknown Laplacian method {method!r}. Available: {list_builders()}")
    N = len(channels) if channels is not None else len(positions)

    W, binfo = GRAPH_BUILDERS[method](
        positions=positions, channels=channels, optodes=optodes, distance=distance,
        eps=eps, k=k, sigma=sigma,
    )
    W = np.asarray(W, dtype=float)

    L = np.asarray(_csgraph_laplacian(W, normed=normed), dtype=float)
    L = 0.5 * (L + L.T)                                   # kill float-drift asymmetry
    if L.shape != (N, N):
        raise AssertionError(f"Laplacian shape {L.shape} != ({N}, {N}) for method {method!r}.")

    n_edges = int((np.triu(W, 1) > 0).sum())
    info = {"method": method, "distance": distance, "normed": bool(normed),
            "n_nodes": int(N), "n_edges": n_edges, **binfo}
    return L, info


def eeg_positions(eeg_chs, eeg_geo3d: xr.DataArray):
    """(N, 3) electrode positions (mm) in channel order, from the recording's
    electrode positions."""
    eeg_geo3d = eeg_geo3d.pint.to("mm").pint.dequantify()
    missing = [str(ch) for ch in eeg_chs if str(ch) not in eeg_geo3d.label.values]
    if missing:
        raise KeyError(f"EEG channels {missing} have no electrode position.")
    return eeg_geo3d.sel(label=[str(ch) for ch in eeg_chs]).values.astype(float)


def fnirs_positions(fnirs_chs, geo3d: xr.DataArray,
                    fnirs_optodes: dict[str, tuple[str, str]]):
    """(N, 3) fNIRS source-detector midpoints (mm) in channel order, from the
    recording's optode positions and the optodes of each channel."""
    geo3d = geo3d.pint.to("mm").pint.dequantify()
    out = []
    for ch in fnirs_chs:
        source, detector = fnirs_optodes[str(ch)]
        src = geo3d.sel(label=source).values.astype(float)
        det = geo3d.sel(label=detector).values.astype(float)
        out.append((src + det) / 2.0)
    return np.asarray(out, dtype=float)


def build_laplacians(spec: dict, eeg_chs, fnirs_chs, geo3d: xr.DataArray,
                     eeg_geo3d: xr.DataArray, fnirs_optodes: dict[str, tuple[str, str]]):
    """
    (Lx, Ly, info) for one subject from a `laplacian` config block.

    Parameters
    ----------
    spec : dict
        {"eeg": {...}, "fnirs": {...}}, each the keyword arguments of
        `build_laplacian`. Empty falls back to `DEFAULT_LAPLACIAN`; a block with one
        modality uses the default for the other.
    eeg_chs, fnirs_chs : sequence of str
        Channel names, in the order of the data the model is fit on.
    geo3d : xr.DataArray
        Optode positions, dims (label, pos), for the fNIRS channels.
    eeg_geo3d : xr.DataArray
        Electrode positions, dims (label, pos), for the EEG channels.
    fnirs_optodes : dict[str, tuple[str, str]]
        {channel: (source, detector)}, labels as in `geo3d`, for every fNIRS channel.

    Returns
    -------
    Lx : np.ndarray
        (n_eeg, n_eeg).
    Ly : np.ndarray
        (n_fnirs, n_fnirs).
    info : dict
        Resolved parameters and node/edge counts per modality.
    """
    eeg_cfg = spec.get("eeg", DEFAULT_LAPLACIAN["eeg"])
    fnirs_cfg = spec.get("fnirs", DEFAULT_LAPLACIAN["fnirs"])
    eeg_chs, fnirs_chs = [str(c) for c in eeg_chs], [str(c) for c in fnirs_chs]
    Lx, info_x = build_laplacian(eeg_positions(eeg_chs, eeg_geo3d), channels=eeg_chs, **eeg_cfg)
    Ly, info_y = build_laplacian(fnirs_positions(fnirs_chs, geo3d, fnirs_optodes),
                                 channels=fnirs_chs,
                                 optodes=[fnirs_optodes[ch] for ch in fnirs_chs], **fnirs_cfg)
    return Lx, Ly, {"eeg": info_x, "fnirs": info_y, "default_used": not spec}
