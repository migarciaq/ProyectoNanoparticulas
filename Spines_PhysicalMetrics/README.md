# nanodot_metrics

Standardized image and physical metrics for the inverse-cycle evaluation of
magnetic nanodot configurations, as introduced in *Image-Driven Estimation of
Magnetic Configurations in Nanodots* (Méndez-Rondón et al.).

The library bundles together the pixel-level, spectral, and physical metrics
used to compare Monte Carlo ground-truth configurations against DDPM-generated
samples, exposing each one as a single, well-documented function so they can be
re-used across experiments without copy-pasting code.

## Installation

```bash
pip install numpy scipy scikit-image
```

Then drop `nanodot_metrics.py` into your project (or `pip install -e .` if you
package it). The module has no other runtime dependencies.

## Conventions

* Images are arrays of the out-of-plane spin component `s_z`, living in
  `[-1, 1]`. Both `(H, W)` and `(H, W, 1)` shapes are accepted throughout the
  library; a trailing channel dimension of size 1 is automatically stripped
  before processing.
* The nanodot is a disk of radius `rd = 18.3 px` centered on the image. All
  physical metrics are restricted to pixels inside this disk.
* Original Monte Carlo images come at `(39, 39)` or `(39, 39, 1)`; the DDPM
  was trained at `(40, 40)`. The helper `resize_original_to_generator` resizes
  the originals bilinearly to match the generator grid, preserving the channel
  dimension when present.
* Every physical-metric function is suffixed with `_orig` (single-configuration
  estimator, applied to the original MC image) or `_gen` (ensemble estimator,
  applied to a stack of `K` generated samples). The math behind each pair is
  not identical — see the *Two regimes* section below.

## A note on preprocessing of the original images

The DDPM is trained on `(40, 40)` images, so the original `(39, 39)` Monte
Carlo configurations are bilinearly upsampled before any comparison. The
earlier prototype additionally applied a `(mn, mx) -&gt; [-1, 1]` renormalisation
using the empirical min and max of a subsample. **That step has been removed
here.** Since the raw MC images already live in `[-1, 1]`, the renormalisation
only injected interpolation noise into the pipeline (boundary pixels of the
disk pick up small attenuation from the bilinear resize, which then made `mn`
and `mx` deviate slightly from `-1` and `+1`). Removing it leaves the
spatial-structure information untouched and keeps `orig` and `gen` on the same
nominal scale.

The bilinear resize itself is preserved (rather than switched to
nearest-neighbour or zero-pad) because the generator was trained against
bilinearly-resized targets; changing the resize at evaluation time would
introduce a domain shift the model never saw.

## Two regimes

The article distinguishes two estimation regimes, matching the two sides of the
comparison:

* **Regime (a) — single-configuration estimators** — suffix `_orig`.
  Each parameter point in the MC dataset corresponds to exactly one
  configuration, so ensemble averages are not directly accessible. Magnetisation
  and energy are evaluated as instantaneous spatial means (which are
  ensemble-unbiased on a single configuration). Susceptibility and specific
  heat are reconstructed via spatial proxies: the static
  fluctuation-dissipation sum rule combined with the Wiener-Khinchin theorem
  for `chi`, and Binder's subsystem fluctuation method for `Cv`. These proxies
  are valid when the correlation length satisfies `xi &lt;&lt; l &lt;&lt; rd`.

* **Regime (b) — ensemble estimators** — suffix `_gen`.
  The DDPM is stochastic and can produce `K` independent samples conditioned on
  the same parameter vector. With an ensemble in hand, `M`, `|M|`, and `E` are
  reported as ensemble means of the per-sample spatial estimators, and `chi`
  and `Cv` come directly from their thermodynamic fluctuation-dissipation
  definitions on the ensemble variance — no spatial proxy is needed.

Whenever both regimes are available, the ensemble estimator on the generated
set is taken as the reference, and the agreement (or disagreement) of the
spatial proxy on the same set serves as an internal calibration for the proxy
used on the simulated set.

## Quick start

```python
import numpy as np
import sys
sys.path.append("/kaggle/input/datasets/carloscanamejoy/physicalmetrics")

from physicalmetrics import (
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

# 1. Original Monte Carlo image at native (39, 39, 1); already in [-1, 1].
img_orig_39 = load_mc_image(params)           # shape (39, 39, 1)

# 2. Resize to the generator grid — output is (40, 40, 1).
img_orig = resize_original_to_generator(img_orig_39)

# 3. Generated samples from the DDPM, conditioned on `params`.
imgs_gen = ddpm.sample_K(params, K=32)        # shape (32, 40, 40, 1), in [-1, 1]
img_gen  = imgs_gen[0]                         # single representative sample (40, 40, 1)

# 4. Image-level metrics on a single pair.
img_m = compute_image_metrics(img_orig, img_gen)
# {"mse", "mse_var", "ssim", "fft_mse", "fft_corr"}

# 5. Physical metrics, regime (a), on the original.
phys_a = compute_physical_metrics_orig(img_orig, T_phys=params["T"])
# {"M", "absM", "chi", "Cv", "E"}

# 6. Physical metrics, regime (b), on the K-sample ensemble.
phys_b = compute_physical_metrics_gen(imgs_gen, T_phys=params["T"])
# {"M", "absM", "chi", "Cv", "E"}

# 7. Or do all of the above in a single call:
all_m = compute_all_metrics(img_orig, imgs_gen, T_phys=params["T"])
# image keys + {"M_orig", "M_gen", "absM_orig", "absM_gen",
#               "chi_orig", "chi_gen", "Cv_orig", "Cv_gen",
#               "E_orig", "E_gen"}
```

## API reference

### Utilities

| Function | Purpose |
| --- | --- |
| `resize_original_to_generator(img, out_size=40)` | Bilinear resize from the MC grid to the DDPM grid, no renormalisation. Accepts `(H, W)` or `(H, W, 1)`; returns the same shape at the target size. |
| `circular_mask(shape, rd=18.3)` | Build the boolean disk mask used by every physical metric. |

### Image metrics

All image-metric functions accept images of shape `(H, W)` or `(H, W, 1)`.

| Function | Returns |
| --- | --- |
| `metric_mse(a, b)` | Pixel-wise mean squared error. |
| `metric_mse_variance(a, b)` | Variance of the per-pixel squared error (Eq. 22). |
| `metric_ssim(a, b, data_range=2.0)` | SSIM with default range matching `[-1, 1]`. |
| `metric_fft(a, b)` | Spectral MSE and Pearson correlation on normalised amplitude spectra (Eqs. 23-24). |
| `compute_image_metrics(a, b)` | All of the above as a single dict. |

### Physical metrics — regime (a): original single configuration

All `_orig` functions accept images of shape `(H, W)` or `(H, W, 1)`.

| Function | Estimator |
| --- | --- |
| `spin_magnetization_orig(img)` | `M = mean(s_z)` inside the disk (Eq. 26). |
| `spin_abs_magnetization_orig(img)` | `\|mean(s_z)\|` (Eq. 28). |
| `spin_susceptibility_orig(img, T)` | FDT sum rule via Wiener-Khinchin, radial cutoff `rd/2` (Eq. 30). |
| `approx_total_energy_orig(img)` | Nearest-neighbour S_z S_z exchange density (Eq. 32). |
| `spin_specific_heat_orig(img, T, block_size=5)` | Binder subsystem fluctuation proxy (Eqs. 34-35). |
| `compute_physical_metrics_orig(img, T_phys)` | All of the above as a single dict. |

### Physical metrics — regime (b): generated ensemble

Each function takes `imgs_K` of shape `(K, H, W)` or `(K, H, W, 1)`, where
the K samples were generated under the same physical parameter vector.

| Function | Estimator |
| --- | --- |
| `spin_magnetization_gen(imgs_K)` | Ensemble mean of per-sample `M` (Eq. 27). |
| `spin_abs_magnetization_gen(imgs_K)` | Ensemble mean of per-sample `\|M\|` (Eq. 29). |
| `spin_susceptibility_gen(imgs_K, T)` | `chi = N * Var(M) / T` over the ensemble (Eq. 31). |
| `approx_total_energy_gen(imgs_K)` | Ensemble mean of per-sample `E` (Eq. 33). |
| `spin_specific_heat_gen(imgs_K, T)` | `Cv = N * Var(E) / T^2` over the ensemble (Eq. 36). |
| `compute_physical_metrics_gen(imgs_K, T_phys)` | All of the above as a single dict. |

### Top-level convenience

| Function | Purpose |
| --- | --- |
| `compute_all_metrics(img_orig, imgs_gen, T_phys)` | Image metrics on `(img_orig, imgs_gen[0])` plus suffixed physical metrics for both regimes, all in one dict. Accepts `(H, W)` or `(H, W, 1)` for `img_orig`, and `(K, H, W)` or `(K, H, W, 1)` for `imgs_gen`. |

## Validity caveats

These caveats apply to the spatial-proxy estimators of regime (a), and are
inherited from the article:

* The `chi` proxy assumes translational invariance and self-averaging inside
  the disk. Near phase transitions, where the correlation length approaches
  `rd`, it underestimates the true thermodynamic `chi`.
* The `Cv` block proxy is valid when `xi &lt;&lt; l &lt;&lt; rd`. With `block_size = 5`
  and `rd = 18.3`, this holds away from critical points; near `T_c` the proxy
  underestimates the peak of `Cv`.
* The exchange energy `E` captures only the diagonal `S_z * S_z` component of
  the full Heisenberg exchange. Transverse components, further-neighbour
  exchanges `J3, J4`, DMI, Zeeman, and single-ion anisotropy are not
  recoverable from the scalar image and are absorbed implicitly through the
  parameter conditioning of the generative model.
* Bilinear resize from `(39, 39)` to `(40, 40)` introduces a small attenuation
  on the ~1-px ring at the disk perimeter. This is a known consequence of
  matching the trained model's input grid.

## Citation

If you use this library, please cite the article:

> Méndez-Rondón et al., *Image-Driven Estimation of Magnetic Configurations in
> Nanodots*, in preparation.

## Acknowledgments

Hermes 62642, Universidad Nacional de Colombia. Project 111908, Minciencias
951-2024.