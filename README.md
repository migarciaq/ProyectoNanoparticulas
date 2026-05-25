# Spines-IA



\# PhysicalMetrics



Standardized image and physical metrics for the inverse-cycle evaluation of

magnetic nanodot configurations, as introduced in \*Image-Driven Estimation of

Magnetic Configurations in Nanodots\* (Méndez-Rondón et al.).



The library bundles together the pixel-level, spectral, and physical metrics

used to compare Monte Carlo ground-truth configurations against DDPM-generated

samples, exposing each one as a single, well-documented function so they can be

re-used across experiments without copy-pasting code.



\## Installation



```bash

pip install numpy scipy scikit-image

```



Then drop `nanodot\_metrics.py` into your project (or `pip install -e .` if you

package it). The module has no other runtime dependencies.



\## Conventions



\* Images are arrays of the out-of-plane spin component `s\_z`, living in

&#x20; `\[-1, 1]`. Both `(H, W)` and `(H, W, 1)` shapes are accepted throughout the

&#x20; library; a trailing channel dimension of size 1 is automatically stripped

&#x20; before processing.

\* The nanodot is a disk of radius `rd = 18.3 px` centered on the image. All

&#x20; physical metrics are restricted to pixels inside this disk.

\* Original Monte Carlo images come at `(39, 39)` or `(39, 39, 1)`; the DDPM

&#x20; was trained at `(40, 40)`. The helper `resize\_original\_to\_generator` resizes

&#x20; the originals bilinearly to match the generator grid, preserving the channel

&#x20; dimension when present.

\* Every physical-metric function is suffixed with `\_orig` (single-configuration

&#x20; estimator, applied to the original MC image) or `\_gen` (ensemble estimator,

&#x20; applied to a stack of `K` generated samples). The math behind each pair is

&#x20; not identical — see the \*Two regimes\* section below.



\## A note on preprocessing of the original images



The DDPM is trained on `(40, 40)` images, so the original `(39, 39)` Monte

Carlo configurations are bilinearly upsampled before any comparison. The

earlier prototype additionally applied a `(mn, mx) -> \[-1, 1]` renormalisation

using the empirical min and max of a subsample. \*\*That step has been removed

here.\*\* Since the raw MC images already live in `\[-1, 1]`, the renormalisation

only injected interpolation noise into the pipeline (boundary pixels of the

disk pick up small attenuation from the bilinear resize, which then made `mn`

and `mx` deviate slightly from `-1` and `+1`). Removing it leaves the

spatial-structure information untouched and keeps `orig` and `gen` on the same

nominal scale.



The bilinear resize itself is preserved (rather than switched to

nearest-neighbour or zero-pad) because the generator was trained against

bilinearly-resized targets; changing the resize at evaluation time would

introduce a domain shift the model never saw.



\## Two regimes



The article distinguishes two estimation regimes, matching the two sides of the

comparison:



\* \*\*Regime (a) — single-configuration estimators\*\* — suffix `\_orig`.

&#x20; Each parameter point in the MC dataset corresponds to exactly one

&#x20; configuration, so ensemble averages are not directly accessible. Magnetisation

&#x20; and energy are evaluated as instantaneous spatial means (which are

&#x20; ensemble-unbiased on a single configuration). Susceptibility and specific

&#x20; heat are reconstructed via spatial proxies: the static

&#x20; fluctuation-dissipation sum rule combined with the Wiener-Khinchin theorem

&#x20; for `chi`, and Binder's subsystem fluctuation method for `Cv`. These proxies

&#x20; are valid when the correlation length satisfies `xi << l << rd`.



\* \*\*Regime (b) — ensemble estimators\*\* — suffix `\_gen`.

&#x20; The DDPM is stochastic and can produce `K` independent samples conditioned on

&#x20; the same parameter vector. With an ensemble in hand, `M`, `|M|`, and `E` are

&#x20; reported as ensemble means of the per-sample spatial estimators, and `chi`

&#x20; and `Cv` come directly from their thermodynamic fluctuation-dissipation

&#x20; definitions on the ensemble variance — no spatial proxy is needed.



Whenever both regimes are available, the ensemble estimator on the generated

set is taken as the reference, and the agreement (or disagreement) of the

spatial proxy on the same set serves as an internal calibration for the proxy

used on the simulated set.



\## Quick start



```python

import numpy as np

import sys

sys.path.append("/kaggle/input/datasets/carloscanamejoy/physicalmetrics")



from physicalmetrics import (

&#x20;       resize\_original\_to\_generator,

&#x20;       # image metrics

&#x20;       metric\_mse, metric\_ssim, metric\_fft,

&#x20;       # physical metrics, regime (a) - original

&#x20;       spin\_magnetization\_orig, spin\_abs\_magnetization\_orig,

&#x20;       spin\_susceptibility\_orig, spin\_specific\_heat\_orig,

&#x20;       approx\_total\_energy\_orig,

&#x20;       # physical metrics, regime (b) - generated ensemble

&#x20;       spin\_magnetization\_gen, spin\_abs\_magnetization\_gen,

&#x20;       spin\_susceptibility\_gen, spin\_specific\_heat\_gen,

&#x20;       approx\_total\_energy\_gen,

&#x20;       # convenience wrappers

&#x20;       compute\_physical\_metrics\_orig,

&#x20;       compute\_physical\_metrics\_gen,

&#x20;       compute\_image\_metrics,

&#x20;   )



\# 1. Original Monte Carlo image at native (39, 39, 1); already in \[-1, 1].

img\_orig\_39 = load\_mc\_image(params)           # shape (39, 39, 1)



\# 2. Resize to the generator grid — output is (40, 40, 1).

img\_orig = resize\_original\_to\_generator(img\_orig\_39)



\# 3. Generated samples from the DDPM, conditioned on `params`.

imgs\_gen = ddpm.sample\_K(params, K=32)        # shape (32, 40, 40, 1), in \[-1, 1]

img\_gen  = imgs\_gen\[0]                         # single representative sample (40, 40, 1)



\# 4. Image-level metrics on a single pair.

img\_m = compute\_image\_metrics(img\_orig, img\_gen)

\# {"mse", "mse\_var", "ssim", "fft\_mse", "fft\_corr"}



\# 5. Physical metrics, regime (a), on the original.

phys\_a = compute\_physical\_metrics\_orig(img\_orig, T\_phys=params\["T"])

\# {"M", "absM", "chi", "Cv", "E"}



\# 6. Physical metrics, regime (b), on the K-sample ensemble.

phys\_b = compute\_physical\_metrics\_gen(imgs\_gen, T\_phys=params\["T"])

\# {"M", "absM", "chi", "Cv", "E"}



\# 7. Or do all of the above in a single call:

all\_m = compute\_all\_metrics(img\_orig, imgs\_gen, T\_phys=params\["T"])

\# image keys + {"M\_orig", "M\_gen", "absM\_orig", "absM\_gen",

\#               "chi\_orig", "chi\_gen", "Cv\_orig", "Cv\_gen",

\#               "E\_orig", "E\_gen"}

```



\## API reference



\### Utilities



| Function | Purpose |

| --- | --- |

| `resize\_original\_to\_generator(img, out\_size=40)` | Bilinear resize from the MC grid to the DDPM grid, no renormalisation. Accepts `(H, W)` or `(H, W, 1)`; returns the same shape at the target size. |

| `circular\_mask(shape, rd=18.3)` | Build the boolean disk mask used by every physical metric. |



\### Image metrics



All image-metric functions accept images of shape `(H, W)` or `(H, W, 1)`.



| Function | Returns |

| --- | --- |

| `metric\_mse(a, b)` | Pixel-wise mean squared error. |

| `metric\_mse\_variance(a, b)` | Variance of the per-pixel squared error (Eq. 22). |

| `metric\_ssim(a, b, data\_range=2.0)` | SSIM with default range matching `\[-1, 1]`. |

| `metric\_fft(a, b)` | Spectral MSE and Pearson correlation on normalised amplitude spectra (Eqs. 23-24). |

| `compute\_image\_metrics(a, b)` | All of the above as a single dict. |



\### Physical metrics — regime (a): original single configuration



All `\_orig` functions accept images of shape `(H, W)` or `(H, W, 1)`.



| Function | Estimator |

| --- | --- |

| `spin\_magnetization\_orig(img)` | `M = mean(s\_z)` inside the disk (Eq. 26). |

| `spin\_abs\_magnetization\_orig(img)` | `\\|mean(s\_z)\\|` (Eq. 28). |

| `spin\_susceptibility\_orig(img, T)` | FDT sum rule via Wiener-Khinchin, radial cutoff `rd/2` (Eq. 30). |

| `approx\_total\_energy\_orig(img)` | Nearest-neighbour S\_z S\_z exchange density (Eq. 32). |

| `spin\_specific\_heat\_orig(img, T, block\_size=5)` | Binder subsystem fluctuation proxy (Eqs. 34-35). |

| `compute\_physical\_metrics\_orig(img, T\_phys)` | All of the above as a single dict. |



\### Physical metrics — regime (b): generated ensemble



Each function takes `imgs\_K` of shape `(K, H, W)` or `(K, H, W, 1)`, where

the K samples were generated under the same physical parameter vector.



| Function | Estimator |

| --- | --- |

| `spin\_magnetization\_gen(imgs\_K)` | Ensemble mean of per-sample `M` (Eq. 27). |

| `spin\_abs\_magnetization\_gen(imgs\_K)` | Ensemble mean of per-sample `\\|M\\|` (Eq. 29). |

| `spin\_susceptibility\_gen(imgs\_K, T)` | `chi = N \* Var(M) / T` over the ensemble (Eq. 31). |

| `approx\_total\_energy\_gen(imgs\_K)` | Ensemble mean of per-sample `E` (Eq. 33). |

| `spin\_specific\_heat\_gen(imgs\_K, T)` | `Cv = N \* Var(E) / T^2` over the ensemble (Eq. 36). |

| `compute\_physical\_metrics\_gen(imgs\_K, T\_phys)` | All of the above as a single dict. |



\### Top-level convenience



| Function | Purpose |

| --- | --- |

| `compute\_all\_metrics(img\_orig, imgs\_gen, T\_phys)` | Image metrics on `(img\_orig, imgs\_gen\[0])` plus suffixed physical metrics for both regimes, all in one dict. Accepts `(H, W)` or `(H, W, 1)` for `img\_orig`, and `(K, H, W)` or `(K, H, W, 1)` for `imgs\_gen`. |



\## Validity caveats



These caveats apply to the spatial-proxy estimators of regime (a), and are

inherited from the article:



\* The `chi` proxy assumes translational invariance and self-averaging inside

&#x20; the disk. Near phase transitions, where the correlation length approaches

&#x20; `rd`, it underestimates the true thermodynamic `chi`.

\* The `Cv` block proxy is valid when `xi << l << rd`. With `block\_size = 5`

&#x20; and `rd = 18.3`, this holds away from critical points; near `T\_c` the proxy

&#x20; underestimates the peak of `Cv`.

\* The exchange energy `E` captures only the diagonal `S\_z \* S\_z` component of

&#x20; the full Heisenberg exchange. Transverse components, further-neighbour

&#x20; exchanges `J3, J4`, DMI, Zeeman, and single-ion anisotropy are not

&#x20; recoverable from the scalar image and are absorbed implicitly through the

&#x20; parameter conditioning of the generative model.

\* Bilinear resize from `(39, 39)` to `(40, 40)` introduces a small attenuation

&#x20; on the \~1-px ring at the disk perimeter. This is a known consequence of

&#x20; matching the trained model's input grid.



\## Citation



If you use this library, please cite the article:



> Méndez-Rondón et al., \*Image-Driven Estimation of Magnetic Configurations in

> Nanodots\*, in preparation.



\## Acknowledgments



Hermes 62642, Universidad Nacional de Colombia. Project 111908, Minciencias

951-2024.

