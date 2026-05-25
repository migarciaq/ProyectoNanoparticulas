"""
nanodot_metrics.py
==================

Standardized image and physical metrics for the inverse-cycle evaluation of
magnetic nanodot configurations (Monte Carlo ground truth vs DDPM-generated).

This module bundles the metrics used in the article "Image-Driven Estimation of
Magnetic Configurations in Nanodots" (Méndez-Rondón et al.) into a single,
dependency-light library. All metrics operate on 2D arrays representing the
out-of-plane spin component ``s_z`` on a square pixel grid.

Conventions
-----------
* Images live in the interval ``[-1, 1]`` (s_z component, normalised).
* The nanodot is a disk of physical radius ``rd = 18.3 px`` centered at the
  image center. Outside the disk, pixels are treated as inactive via a circular
  mask.
* Original Monte Carlo configurations come at ``(39, 39)`` and must be resized
  to ``(40, 40)`` to match the DDPM output. The library exposes
  :func:`resize_original_to_generator` for that purpose. It uses bilinear
  interpolation (consistent with the notebook). It does **not** perform the
  ``(mn, mx) -> [-1, 1]`` renormalisation: since the raw MC images already live
  in ``[-1, 1]``, that step only injected interpolation noise into the
  comparison.
* Functions are named with the suffixes ``_orig`` and ``_gen`` to make the
  side of the comparison explicit. The underlying math is identical for both;
  the suffix is a semantic tag, not a different algorithm.
* Two regimes are supported, matching the article (Section 3 / Eqs. 26-36):

  - **Regime (a) — single-configuration estimators.** Used for the original
    Monte Carlo images, where only one configuration exists per parameter
    point. Magnetisation and energy are evaluated as instantaneous spatial
    means; susceptibility and specific heat are reconstructed via spatial
    proxies (FDT sum rule with Wiener-Khinchin for chi; Binder's subsystem
    fluctuation method for Cv).
  - **Regime (b) — ensemble estimators.** Used for DDPM-generated images,
    where K independent samples per parameter point are available. M, |M|, E
    are ensemble means; chi and Cv come directly from the
    fluctuation-dissipation definitions on the ensemble variance.

Dependencies
------------
numpy, scipy, scikit-image.

Typical usage
-------------
::

    from nanodot_metrics import (
        resize_original_to_generator,
        # image metrics
        metric_mse, metric_ssim, metric_fft,
        # physical metrics, regime (a) - original
        spin_magnetization_orig, spin_abs_magnetization_orig,
        spin_susceptibility_orig, spin_specific_heat_orig,
        approx_total_energy_orig,
        # physical metrics, regime (b) - generated ensemble
        spin_magnetization_gen, spin_abs_magnetization_gen,
        spin_susceptibility_gen, spin_specific_heat_gen,
        approx_total_energy_gen,
        # convenience wrappers
        compute_physical_metrics_orig,
        compute_physical_metrics_gen,
        compute_image_metrics,
    )

    img_orig_40 = resize_original_to_generator(img_orig_39)
    img_gen_40  = ddpm.sample(params)                # (40, 40), already in [-1, 1]
    imgs_gen_K  = ddpm.sample_K(params, K=32)        # (K, 40, 40)

    phys_a = compute_physical_metrics_orig(img_orig_40, T_phys=T)
    phys_b = compute_physical_metrics_gen(imgs_gen_K,  T_phys=T)
    img_m  = compute_image_metrics(img_orig_40, img_gen_40)

References
----------
* Mohylna & Žukovič, 2022, Eqs. 1-4.
* Newman & Barkema, 1999, Sec. 3.7 (FDT sum rule for chi).
* Binder, 1981; Landau & Binder, 2014, Ch. 4 (subsystem fluctuation method).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from skimage.metrics import structural_similarity as _ssim_fn


# =============================================================================
# Constants
# =============================================================================

#: Physical radius of the simulated nanodot, in pixels on the 40x40 grid.
RD_PIXELS: float = 18.3

#: Default radial cutoff for the chi spatial proxy (Wiener-Khinchin truncation).
#: Cutoff at rd/2 avoids the Parseval trivialisation that occurs when summing
#: G(r) over the whole grid.
DEFAULT_CHI_R_CUTOFF: float = RD_PIXELS / 2.0

#: Default block size (in pixels) for the Binder subsystem proxy of Cv.
DEFAULT_BLOCK_SIZE: int = 5

#: Target grid size of the generator (DDPM output).
GENERATOR_IMG_SIZE: int = 40

#: Native grid size of the Monte Carlo dataset.
DATASET_IMG_SIZE: int = 39


# =============================================================================
# Resize utility
# =============================================================================

def resize_original_to_generator(
    img: np.ndarray,
    out_size: int = GENERATOR_IMG_SIZE,
) -> np.ndarray:
    """Bilinear resize a Monte Carlo image to match the generator grid.

    The DDPM was trained on ``(40, 40)`` images, while the raw Monte Carlo
    dataset is ``(39, 39)``. This function performs the bilinear resize used
    throughout the inverse-cycle evaluation, without applying any
    ``(mn, mx) -> [-1, 1]`` renormalisation. Since the input is already in
    ``[-1, 1]``, the renormalisation present in the original notebook only
    added interpolation noise.

    Parameters
    ----------
    img : ndarray of shape (H, W) or (H, W, 1)
        Spin image in ``[-1, 1]``. A trailing channel dimension of size 1
        is accepted and preserved in the output.
    out_size : int, default ``GENERATOR_IMG_SIZE`` (40)
        Target side length of the output square grid.

    Returns
    -------
    ndarray of shape (out_size, out_size) or (out_size, out_size, 1)
        Resized image, still in ``[-1, 1]``. The output shape matches the
        input: ``(H, W)`` inputs return ``(out_size, out_size)``; ``(H, W, 1)``
        inputs return ``(out_size, out_size, 1)``.

    Notes
    -----
    Bilinear interpolation is preserved (rather than nearest neighbour) for
    consistency with the trained model and prior experimental runs. Bilinear
    interpolation does smooth boundary pixels of the disk; this introduces a
    small attenuation of extreme spin values near the perimeter, but the
    effect is bounded to a ~1-pixel-thick ring and is documented as a known
    limitation in the article.
    """
    img = np.asarray(img, dtype=np.float64)

    # Accept (H, W, 1) by squeezing the channel dimension before processing.
    has_channel = img.ndim == 3 and img.shape[2] == 1
    if has_channel:
        img = img[:, :, 0]

    if img.ndim != 2:
        raise ValueError(
            f"Expected image of shape (H, W) or (H, W, 1), got shape {img.shape}."
        )

    h, w = img.shape
    if (h, w) == (out_size, out_size):
        result = img.astype(np.float64)
    else:
        zoom_factors = (out_size / h, out_size / w)
        # order=1 -> bilinear; mode='reflect' keeps the disk centred without
        # spurious boundary artefacts.
        result = zoom(img, zoom_factors, order=1, mode="reflect", grid_mode=False)

    # Restore the channel dimension if the input had one.
    if has_channel:
        result = result[:, :, np.newaxis]
    return result


# =============================================================================
# Geometry helpers
# =============================================================================

def circular_mask(
    shape: tuple[int, int],
    rd: float = RD_PIXELS,
) -> np.ndarray:
    """Build the circular nanodot mask centred on the image.

    Parameters
    ----------
    shape : tuple of int
        Image shape ``(H, W)``.
    rd : float, default :data:`RD_PIXELS`
        Disk radius in pixels.

    Returns
    -------
    ndarray of bool with shape ``(H, W)``
        ``True`` for pixels inside the disk.
    """
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= rd ** 2


def _validate_image(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected image of shape (H, W) or (H, W, 1), got shape {arr.shape}."
        )
    return arr


def _validate_stack(imgs_K: np.ndarray) -> np.ndarray:
    arr = np.asarray(imgs_K, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[:, :, :, 0]
    if arr.ndim != 3:
        raise ValueError(
            f"Expected stack of shape (K, H, W) or (K, H, W, 1); got shape {arr.shape}."
        )
    if arr.shape[0] < 1:
        raise ValueError("Empty ensemble: K must be >= 1.")
    return arr


# =============================================================================
# Image metrics (pixel- and spectrum-level)
# =============================================================================

def metric_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images.

    Both images are cast to ``float64`` before subtraction. No mask is
    applied: this is a raw pixel-wise comparison over the full grid.
    """
    a = _validate_image(a)
    b = _validate_image(b)
    return float(np.mean((a - b) ** 2))


def metric_mse_variance(a: np.ndarray, b: np.ndarray) -> float:
    """Variance of the per-pixel squared error.

    Implements Eq. 22 in the article. High values indicate that the error is
    concentrated on a few pixels (sharp localised mismatch) rather than
    spread uniformly.
    """
    a = _validate_image(a)
    b = _validate_image(b)
    sq_err = (a - b) ** 2
    return float(np.var(sq_err))


def metric_ssim(a: np.ndarray, b: np.ndarray, data_range: float = 2.0) -> float:
    """Structural similarity index.

    The default ``data_range=2.0`` reflects images living in ``[-1, 1]``.

    Parameters
    ----------
    a, b : ndarray of shape (H, W)
        Images to compare.
    data_range : float, default 2.0
        Dynamic range of the input images.

    Returns
    -------
    float
        SSIM in ``[-1, 1]``; 1.0 means structural identity.
    """
    a = _validate_image(a)
    b = _validate_image(b)
    return float(_ssim_fn(a, b, data_range=data_range))


def _normalised_spectrum(img: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(f)
    return mag / (mag.max() + 1e-12)


def metric_fft(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Spectral comparison metrics (Eqs. 23-24 in the article).

    Returns a dictionary with:

    - ``"mse"`` : mean squared error between the normalised amplitude spectra.
    - ``"corr"`` : Pearson correlation between the normalised amplitude spectra.
    """
    a = _validate_image(a)
    b = _validate_image(b)
    sa, sb = _normalised_spectrum(a), _normalised_spectrum(b)
    fft_mse = float(np.mean((sa - sb) ** 2))
    fft_corr = float(np.corrcoef(sa.ravel(), sb.ravel())[0, 1])
    return {"mse": fft_mse, "corr": fft_corr}


def compute_image_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Convenience wrapper: compute all image metrics on a single pair.

    Returns
    -------
    dict
        ``{"mse", "mse_var", "ssim", "fft_mse", "fft_corr"}``.
    """
    fft_d = metric_fft(a, b)
    return {
        "mse":      metric_mse(a, b),
        "mse_var":  metric_mse_variance(a, b),
        "ssim":     metric_ssim(a, b),
        "fft_mse":  fft_d["mse"],
        "fft_corr": fft_d["corr"],
    }


# =============================================================================
# Physical metrics - core single-configuration kernels (private)
# =============================================================================
# These kernels implement the spatial-proxy estimators of regime (a). They are
# called both by the *_orig functions (where they ARE the estimator) and by the
# *_gen functions (which average them over an ensemble of K samples).

def _M_single(img: np.ndarray, mask: np.ndarray) -> float:
    """Spatial magnetisation (signed) of a single configuration."""
    return float(img[mask].mean())


def _absM_single(img: np.ndarray, mask: np.ndarray) -> float:
    """Spatial magnetisation (absolute) of a single configuration."""
    return float(abs(img[mask].mean()))


def _chi_proxy_single(
    img: np.ndarray,
    mask: np.ndarray,
    T: float,
    r_cutoff: float,
) -> float:
    """Spatial proxy of chi via the FDT sum rule and Wiener-Khinchin (Eq. 30).

    chi = (1/T) * sum_{|r| <= r_cutoff} G(r),
    with G(r) the connected spatial correlation function computed from a
    single configuration via Wiener-Khinchin.
    """
    if T == 0:
        return float("nan")
    s = img * mask
    n_m = int(mask.sum())
    sbar = s.sum() / n_m
    s_c = (img - sbar) * mask
    f = np.fft.fft2(s_c)
    g_full = np.real(np.fft.ifft2(np.abs(f) ** 2)) / n_m
    h, w = img.shape
    dys = np.arange(h); dys = np.where(dys > h // 2, dys - h, dys)
    dxs = np.arange(w); dxs = np.where(dxs > w // 2, dxs - w, dxs)
    dy, dx = np.meshgrid(dys, dxs, indexing="ij")
    r = np.sqrt(dy ** 2 + dx ** 2)
    cutoff_mask = (r <= r_cutoff)
    return float(g_full[cutoff_mask].sum() / T)


def _E_single(img: np.ndarray, mask: np.ndarray) -> float:
    """Nearest-neighbour exchange energy density (Eq. 32) of a single config.

    Captures only the diagonal S_z * S_z component of the full Heisenberg
    exchange. Pairs crossing the disk boundary are excluded.
    """
    s = img * mask
    pair_down = s * np.roll(s, -1, axis=0)
    pair_rght = s * np.roll(s, -1, axis=1)
    m_down = mask & np.roll(mask, -1, axis=0)
    m_rght = mask & np.roll(mask, -1, axis=1)
    pair_sum = pair_down[m_down].sum() + pair_rght[m_rght].sum()
    n_m = int(mask.sum())
    return float(-pair_sum / n_m)


def _Cv_proxy_single(
    img: np.ndarray,
    mask: np.ndarray,
    T: float,
    block_size: int,
) -> float:
    """Binder subsystem proxy of Cv (Eqs. 34-35) on a single configuration.

    Cv = |B| / T^2 * Var_blocks(eps_k), where eps_k is the local energy
    density on the k-th non-overlapping block of side block_size. Only fully
    inside-disk blocks are retained.
    """
    if T == 0:
        return float("nan")
    s = img * mask
    h, w = s.shape
    pair_down = s * np.roll(s, -1, axis=0)
    pair_rght = s * np.roll(s, -1, axis=1)

    eps_list: list[float] = []
    for r in range(0, h - block_size + 1, block_size):
        for c in range(0, w - block_size + 1, block_size):
            blk_mask = np.zeros((h, w), dtype=bool)
            blk_mask[r:r + block_size, c:c + block_size] = True
            roi = blk_mask & mask
            if roi.sum() != block_size * block_size:
                continue
            roi_down = roi & np.roll(roi, -1, axis=0)
            roi_rght = roi & np.roll(roi, -1, axis=1)
            pair_sum_blk = pair_down[roi_down].sum() + pair_rght[roi_rght].sum()
            eps_k = -pair_sum_blk / (block_size * block_size)
            eps_list.append(eps_k)
    if len(eps_list) < 2:
        return float("nan")
    eps = np.asarray(eps_list, dtype=np.float64)
    b_size = block_size * block_size
    var_eps = float(eps.var(ddof=0))
    return float(b_size * var_eps / (T ** 2))


# =============================================================================
# Physical metrics - regime (a): ORIGINAL (single Monte Carlo configuration)
# =============================================================================
# Suffix _orig. These are single-configuration estimators applied to the
# resized 40x40 MC image. M, |M|, E are instantaneous spatial means
# (ensemble-unbiased estimators of <.> for a single config). chi and Cv are
# spatial proxies that approximate the thermodynamic definition under the
# assumption xi << l << rd.

def spin_magnetization_orig(
    img: np.ndarray,
    rd: float = RD_PIXELS,
) -> float:
    """Magnetisation M on a single original configuration (Eq. 26).

    M = (1/|D|) * sum_{i in D} s_z(i), with D the disk mask.

    Returns
    -------
    float in [-1, 1]
        M ~ +/-1 for uniformly polarised states; M ~ 0 for compensated,
        antiferromagnetic, or chiral textures (e.g. skyrmion lattice).
    """
    img = _validate_image(img)
    mask = circular_mask(img.shape, rd=rd)
    return _M_single(img, mask)


def spin_abs_magnetization_orig(
    img: np.ndarray,
    rd: float = RD_PIXELS,
) -> float:
    """Absolute magnetisation |M| on a single original configuration (Eq. 28).

    Orientation-invariant order parameter, appropriate when inverted domains
    may coexist.
    """
    img = _validate_image(img)
    mask = circular_mask(img.shape, rd=rd)
    return _absM_single(img, mask)


def spin_susceptibility_orig(
    img: np.ndarray,
    T: float,
    rd: float = RD_PIXELS,
    r_cutoff: float | None = None,
) -> float:
    """Susceptibility chi on a single original configuration (Eq. 30).

    Spatial proxy via the static fluctuation-dissipation sum rule combined
    with the Wiener-Khinchin theorem.

    Parameters
    ----------
    img : ndarray of shape (H, W)
    T : float
        Physical temperature in units of ``J / k_B``. Returns NaN if ``T == 0``.
    rd : float, default :data:`RD_PIXELS`
    r_cutoff : float, optional
        Radial cutoff for the sum over G(r). Defaults to
        :data:`DEFAULT_CHI_R_CUTOFF` (``rd / 2``) to avoid Parseval
        trivialisation.

    Notes
    -----
    The proxy underestimates the true thermodynamic chi near phase transitions,
    where the correlation length approaches the system size.
    """
    img = _validate_image(img)
    mask = circular_mask(img.shape, rd=rd)
    if r_cutoff is None:
        r_cutoff = DEFAULT_CHI_R_CUTOFF
    return _chi_proxy_single(img, mask, T=T, r_cutoff=r_cutoff)


def approx_total_energy_orig(
    img: np.ndarray,
    rd: float = RD_PIXELS,
) -> float:
    """Nearest-neighbour exchange energy density E on a single original (Eq. 32).

    Only the diagonal S_z * S_z component is reconstructible from the scalar
    image. Further-neighbour exchanges, DMI, Zeeman and anisotropy terms are
    absorbed implicitly through the parameter conditioning of the generative
    model.
    """
    img = _validate_image(img)
    mask = circular_mask(img.shape, rd=rd)
    return _E_single(img, mask)


def spin_specific_heat_orig(
    img: np.ndarray,
    T: float,
    rd: float = RD_PIXELS,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> float:
    """Specific heat Cv on a single original configuration (Eqs. 34-35).

    Binder subsystem fluctuation method. Returns NaN if ``T == 0`` or if
    fewer than 2 fully-inside-disk blocks are available.

    Parameters
    ----------
    block_size : int, default :data:`DEFAULT_BLOCK_SIZE` (5)
        Side of the square blocks (in pixels). For a 40x40 image with
        ``rd = 18.3``, the default yields ~50 valid blocks.
    """
    img = _validate_image(img)
    mask = circular_mask(img.shape, rd=rd)
    return _Cv_proxy_single(img, mask, T=T, block_size=block_size)


def compute_physical_metrics_orig(
    img: np.ndarray,
    T_phys: float,
    rd: float = RD_PIXELS,
    r_cutoff_chi: float | None = None,
    block_size_cv: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, float]:
    """Compute all regime (a) metrics on a single original configuration.

    Returns
    -------
    dict
        ``{"M", "absM", "chi", "Cv", "E"}``.
    """
    img = _validate_image(img)
    mask = circular_mask(img.shape, rd=rd)
    if r_cutoff_chi is None:
        r_cutoff_chi = DEFAULT_CHI_R_CUTOFF
    return {
        "M":    _M_single(img, mask),
        "absM": _absM_single(img, mask),
        "chi":  _chi_proxy_single(img, mask, T=T_phys, r_cutoff=r_cutoff_chi),
        "Cv":   _Cv_proxy_single(img, mask, T=T_phys, block_size=block_size_cv),
        "E":    _E_single(img, mask),
    }


# =============================================================================
# Physical metrics - regime (b): GENERATED (ensemble of DDPM samples)
# =============================================================================
# Suffix _gen. Each function takes a stack imgs_K of shape (K, H, W) of K
# images independently generated from the SAME physical parameter vector.
# M, |M|, E are ensemble means. chi and Cv use the thermodynamic
# fluctuation-dissipation definition on the ensemble variance, NOT the spatial
# proxy. Whenever K >= 2, these are the reference estimators.

def spin_magnetization_gen(
    imgs_K: np.ndarray,
    rd: float = RD_PIXELS,
) -> float:
    """Ensemble magnetisation <M> on K generated configurations (Eq. 27).

    Parameters
    ----------
    imgs_K : ndarray of shape (K, H, W)
        Stack of K images generated under the same parameter vector.
    """
    arr = _validate_stack(imgs_K)
    mask = circular_mask(arr.shape[1:], rd=rd)
    n_m = int(mask.sum())
    M_per_sample = arr[:, mask].sum(axis=1) / n_m
    return float(M_per_sample.mean())


def spin_abs_magnetization_gen(
    imgs_K: np.ndarray,
    rd: float = RD_PIXELS,
) -> float:
    """Ensemble absolute magnetisation <|M|> on K generated configs (Eq. 29).

    Note that ``<|M|> != |<M>|`` in general; the absolute value is taken
    before averaging over the ensemble.
    """
    arr = _validate_stack(imgs_K)
    mask = circular_mask(arr.shape[1:], rd=rd)
    n_m = int(mask.sum())
    M_per_sample = arr[:, mask].sum(axis=1) / n_m
    return float(np.abs(M_per_sample).mean())


def spin_susceptibility_gen(
    imgs_K: np.ndarray,
    T: float,
    rd: float = RD_PIXELS,
) -> float:
    """Ensemble susceptibility chi on K generated configurations (Eq. 31).

    Direct thermodynamic estimator:

        chi = (N / T) * Var_ens(M),

    with ``N`` the number of in-disk sites and ``M`` the per-sample spatial
    magnetisation. Returns NaN if ``T == 0`` or ``K < 2``.
    """
    arr = _validate_stack(imgs_K)
    if T == 0 or arr.shape[0] < 2:
        return float("nan")
    mask = circular_mask(arr.shape[1:], rd=rd)
    n_m = int(mask.sum())
    M_per_sample = arr[:, mask].sum(axis=1) / n_m
    var_M = float(M_per_sample.var(ddof=0))
    return float(n_m * var_M / T)


def approx_total_energy_gen(
    imgs_K: np.ndarray,
    rd: float = RD_PIXELS,
) -> float:
    """Ensemble exchange energy density <E> on K generated configs (Eq. 33)."""
    arr = _validate_stack(imgs_K)
    mask = circular_mask(arr.shape[1:], rd=rd)
    K = arr.shape[0]
    E_per_sample = np.empty(K, dtype=np.float64)
    for k in range(K):
        E_per_sample[k] = _E_single(arr[k], mask)
    return float(E_per_sample.mean())


def spin_specific_heat_gen(
    imgs_K: np.ndarray,
    T: float,
    rd: float = RD_PIXELS,
) -> float:
    """Ensemble specific heat Cv on K generated configurations (Eq. 36).

    Direct thermodynamic estimator:

        Cv = (N / T^2) * Var_ens(E),

    with ``E`` the per-sample exchange energy density. Returns NaN if
    ``T == 0`` or ``K < 2``.
    """
    arr = _validate_stack(imgs_K)
    if T == 0 or arr.shape[0] < 2:
        return float("nan")
    mask = circular_mask(arr.shape[1:], rd=rd)
    n_m = int(mask.sum())
    K = arr.shape[0]
    E_per_sample = np.empty(K, dtype=np.float64)
    for k in range(K):
        E_per_sample[k] = _E_single(arr[k], mask)
    var_E = float(E_per_sample.var(ddof=0))
    return float(n_m * var_E / (T ** 2))


def compute_physical_metrics_gen(
    imgs_K: np.ndarray,
    T_phys: float,
    rd: float = RD_PIXELS,
) -> dict[str, float]:
    """Compute all regime (b) metrics on an ensemble of K generated images.

    Returns
    -------
    dict
        ``{"M", "absM", "chi", "Cv", "E"}``.
    """
    arr = _validate_stack(imgs_K)
    mask = circular_mask(arr.shape[1:], rd=rd)
    n_m = int(mask.sum())
    K = arr.shape[0]

    M_per_sample = arr[:, mask].sum(axis=1) / n_m
    E_per_sample = np.empty(K, dtype=np.float64)
    for k in range(K):
        E_per_sample[k] = _E_single(arr[k], mask)

    if T_phys == 0 or K < 2:
        chi = float("nan")
        Cv = float("nan")
    else:
        chi = float(n_m * float(M_per_sample.var(ddof=0)) / T_phys)
        Cv = float(n_m * float(E_per_sample.var(ddof=0)) / (T_phys ** 2))

    return {
        "M":    float(M_per_sample.mean()),
        "absM": float(np.abs(M_per_sample).mean()),
        "chi":  chi,
        "Cv":   Cv,
        "E":    float(E_per_sample.mean()),
    }


# =============================================================================
# Top-level convenience: compare original vs generated in a single call
# =============================================================================

def compute_all_metrics(
    img_orig: np.ndarray,
    imgs_gen: np.ndarray,
    T_phys: float,
    rd: float = RD_PIXELS,
    r_cutoff_chi: float | None = None,
    block_size_cv: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, float]:
    """Compute the full set of metrics for a single (orig, gen) comparison.

    Parameters
    ----------
    img_orig : ndarray of shape (H, W)
        Original Monte Carlo image, already resized to the generator grid.
    imgs_gen : ndarray of shape (H, W) or (K, H, W)
        Generated image(s). A 2D input is treated as ``K = 1`` and the
        regime (b) chi/Cv will be NaN. A 3D input is treated as an ensemble.
    T_phys : float
        Physical temperature for chi and Cv. Pass the input parameter value
        used to generate the ensemble.
    rd, r_cutoff_chi, block_size_cv : see the individual functions.

    Returns
    -------
    dict
        Keys: image metrics (``mse``, ``mse_var``, ``ssim``, ``fft_mse``,
        ``fft_corr``) plus suffixed physical metrics (``M_orig``, ``M_gen``,
        ``absM_orig``, ``absM_gen``, ``chi_orig``, ``chi_gen``, ``Cv_orig``,
        ``Cv_gen``, ``E_orig``, ``E_gen``). The image metrics use the first
        generated sample as the representative when an ensemble is passed.
    """
    img_orig = _validate_image(img_orig)
    imgs_gen_arr = np.asarray(imgs_gen, dtype=np.float64)
    # Strip trailing channel dimension: (H,W,1) -> (H,W) and (K,H,W,1) -> (K,H,W)
    if imgs_gen_arr.ndim == 3 and imgs_gen_arr.shape[2] == 1:
        imgs_gen_arr = imgs_gen_arr[:, :, 0]
    elif imgs_gen_arr.ndim == 4 and imgs_gen_arr.shape[3] == 1:
        imgs_gen_arr = imgs_gen_arr[:, :, :, 0]
    if imgs_gen_arr.ndim == 2:
        imgs_gen_stack = imgs_gen_arr[None, ...]
        img_gen_repr = imgs_gen_arr
    elif imgs_gen_arr.ndim == 3:
        imgs_gen_stack = imgs_gen_arr
        img_gen_repr = imgs_gen_arr[0]
    else:
        raise ValueError(
            f"imgs_gen must be 2D (H,W), 3D (K,H,W), or include a trailing "
            f"channel dimension of 1; got shape {imgs_gen_arr.shape}."
        )

    out: dict[str, float] = {}
    out.update(compute_image_metrics(img_orig, img_gen_repr))

    phys_orig = compute_physical_metrics_orig(
        img_orig, T_phys=T_phys, rd=rd,
        r_cutoff_chi=r_cutoff_chi, block_size_cv=block_size_cv,
    )
    phys_gen = compute_physical_metrics_gen(
        imgs_gen_stack, T_phys=T_phys, rd=rd,
    )
    out.update({f"{k}_orig": v for k, v in phys_orig.items()})
    out.update({f"{k}_gen":  v for k, v in phys_gen.items()})
    return out


__all__ = [
    # constants
    "RD_PIXELS",
    "DEFAULT_CHI_R_CUTOFF",
    "DEFAULT_BLOCK_SIZE",
    "GENERATOR_IMG_SIZE",
    "DATASET_IMG_SIZE",
    # utilities
    "resize_original_to_generator",
    "circular_mask",
    # image metrics
    "metric_mse",
    "metric_mse_variance",
    "metric_ssim",
    "metric_fft",
    "compute_image_metrics",
    # physical metrics - regime (a)
    "spin_magnetization_orig",
    "spin_abs_magnetization_orig",
    "spin_susceptibility_orig",
    "approx_total_energy_orig",
    "spin_specific_heat_orig",
    "compute_physical_metrics_orig",
    # physical metrics - regime (b)
    "spin_magnetization_gen",
    "spin_abs_magnetization_gen",
    "spin_susceptibility_gen",
    "approx_total_energy_gen",
    "spin_specific_heat_gen",
    "compute_physical_metrics_gen",
    # convenience
    "compute_all_metrics",
]
