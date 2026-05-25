# Theory and methodology

This document is the technical companion to `nanodot_metrics.py`. It collects
the formal definitions, derivations, validity assumptions, and bibliographic
references behind every metric exposed by the library, so that each function
call is traceable to an equation in the physics literature.

It is organised by metric family, in the same order as the API:

1. [Conventions and notation](#1-conventions-and-notation)
2. [Preprocessing: 39 to 40 resize](#2-preprocessing-39-to-40-resize)
3. [Image metrics](#3-image-metrics)
4. [Physical metrics &mdash; regime (a): single configuration](#4-physical-metrics--regime-a-single-configuration)
5. [Physical metrics &mdash; regime (b): ensemble](#5-physical-metrics--regime-b-ensemble)
6. [Why two regimes](#6-why-two-regimes)
7. [Limitations and validity windows](#7-limitations-and-validity-windows)
8. [References](#8-references)

The notation throughout matches Section 3 of the article *Image-Driven
Estimation of Magnetic Configurations in Nanodots* (Méndez-Rondón et al.).

---

## 1. Conventions and notation

* The simulated lattice is a magnetic nanodot. The spin field is defined
  only inside a circular disk of physical radius $r_d = 18.3$ px on the
  $40 \times 40$ pixel grid, and is set to zero outside.
* Each pixel $i$ in the disk carries the out-of-plane component
  $s_z(i) \in [-1, 1]$ of the normalised spin vector $\mathbf{s}_i$. The full
  spin vector $\mathbf{s}_i \in \mathbb{R}^3$ with $\|\mathbf{s}_i\|_2 = 1$ is
  not reconstructible from the scalar image; only $s_z$ is observable.
* The disk mask is

$$
\mathcal{D} = \big\\{(y, x) \in \mathbb{Z}^2 : (y - c_y)^2 + (x - c_x)^2 \le r_d^2\big\\},
$$

with $(c_y, c_x) = \big(\tfrac{H-1}{2}, \tfrac{W-1}{2}\big)$ the image centre
and $N = |\mathcal{D}|$ the number of active pixels (approximately 1051 for
$40 \times 40$ with $r_d = 18.3$). All physical metrics are restricted to
$i \in \mathcal{D}$.

* The temperature $T$ that enters $\chi$ and $C_v$ is the physical
  temperature in units of $J / k_B$, where $J$ is the first-neighbour
  exchange. The library takes $T$ directly from column 0 of the dataset
  `params` array.

---

## 2. Preprocessing: 39 to 40 resize

The Monte Carlo dataset is stored at native resolution $(39, 39)$, while the
DDPM is trained at $(40, 40)$. Comparing the two requires a common grid; the
choice is between resizing the originals up or downsampling the generator
output. Since the model was trained against $(40, 40)$ targets, downsampling
the generator at inference time would introduce a domain shift the model never
saw. Instead, the originals are bilinearly upsampled to $(40, 40)$ before any
metric is computed.

An earlier prototype additionally applied a per-batch renormalisation

$$
\tilde{x} \;\leftarrow\; \frac{x - m_{\min}}{m_{\max} - m_{\min}} \cdot 2 - 1,
$$

with $m_{\min}, m_{\max}$ the empirical minimum and maximum of a subsample of
the resized dataset. **This renormalisation is omitted in this library.**
Since the raw Monte Carlo images already live in $[-1, 1]$, the step only
injected interpolation noise: the bilinear resize slightly attenuates spin
values on the ~1-px ring at the disk perimeter, which pushes $m_{\min}$ and
$m_{\max}$ away from $\pm 1$, and the subsequent affine rescaling then
distorted the comparison without any physical justification. Removing it
keeps `orig` and `gen` on the same nominal scale.

The bilinear interpolation itself remains a documented source of attenuation
on the disk boundary (Section 7).

---

## 3. Image metrics

These metrics operate on the raw pixel arrays and do not use the disk mask.
They serve as low-level fidelity indicators.

### 3.1 Mean squared error

For two images $a, b \in \mathbb{R}^{H \times W}$ with $N_p = H \cdot W$
pixels,

$$
\mathrm{MSE}(a, b) = \frac{1}{N_p} \sum_{i=1}^{N_p} (a_i - b_i)^2.
$$

### 3.2 Variance of the per-pixel squared error

$$
\mathrm{Var\\_MSE}(a, b)
= \frac{1}{N_p} \sum_{i=1}^{N_p} \big(e_i^2 - \overline{e^2}\big)^2,
\qquad
e_i = (a_i - b_i)^2, \qquad
\overline{e^2} = \frac{1}{N_p} \sum_i e_i.
$$

A high value indicates that the squared error is concentrated on a few pixels
(sharp localised mismatch) rather than spread uniformly. Used to distinguish
"globally wrong" from "locally wrong" reconstructions [^11].

### 3.3 Structural similarity index (SSIM)

The library calls the standard implementation in `scikit-image` with the
single-image data range set to $2.0$ (matching $[-1, 1]$):

$$
\mathrm{SSIM}(a, b) = \frac{(2 \mu_a \mu_b + C_1)(2 \sigma_{ab} + C_2)}
                          {(\mu_a^2 + \mu_b^2 + C_1)(\sigma_a^2 + \sigma_b^2 + C_2)},
$$

with local statistics computed under a sliding Gaussian window. Original
formulation in Wang et al. [^16]; the article uses $1 - \mathrm{SSIM}$ as a
loss term (Eq. 17 of the article).

### 3.4 Spectral metrics (FFT-MSE, FFT-Corr)

Define the normalised amplitude spectrum

$$
\tilde{S}_x = \frac{|\mathcal{F}\\{x\\}|}{\max |\mathcal{F}\\{x\\}|},
\qquad \mathcal{F}\\{x\\} \in \mathbb{C}^{H \times W} \text{ centred at zero frequency}.
$$

Then

$$
\mathrm{FFT\\_MSE}(a, b)
= \frac{1}{N_p} \sum_{i=1}^{N_p} \big(\tilde{S}_a^{(i)} - \tilde{S}_b^{(i)}\big)^2,
$$

and

$$
\mathrm{FFT\\_Corr}(a, b)
= \frac{\sum_i \big(\tilde{S}_a^{(i)} - \bar{\tilde{S}}_a\big)\big(\tilde{S}_b^{(i)} - \bar{\tilde{S}}_b\big)}
       {\sqrt{\sum_i \big(\tilde{S}_a^{(i)} - \bar{\tilde{S}}_a\big)^2}
        \sqrt{\sum_i \big(\tilde{S}_b^{(i)} - \bar{\tilde{S}}_b\big)^2}}.
$$

Spectral comparison is robust to spatial phase shifts: two configurations
that are translates of each other (or differ only by the position of the
skyrmion core, for example) have nearly identical amplitude spectra and would
score very differently under pixel-wise MSE. This is the metric the article
recommends for comparing periodic magnetic textures whose absolute pixel
location is energetically degenerate.

---

## 4. Physical metrics &mdash; regime (a): single configuration

Each parameter point in the Monte Carlo dataset produces exactly one
configuration. Ensemble averages $\langle \cdot \rangle$ in the
fluctuation-dissipation sense are not directly accessible. The library
therefore uses two distinct strategies:

* **Magnetisation $M$ and exchange energy density $E$**: spatial mean over
  the disk. This is an ensemble-unbiased estimator of
  $\langle M \rangle$ and $\langle E \rangle$ on a single configuration.
* **Susceptibility $\chi$ and specific heat $C_v$**: spatial proxies that
  exploit the static fluctuation-dissipation theorem (FDT) sum rule for
  $\chi$ and Binder's subsystem fluctuation method for $C_v$.

### 4.1 Magnetisation $M$ (Eq. 26)

$$
M = \frac{1}{N} \sum_{i \in \mathcal{D}} s_z(i).
$$

The spatial mean is an ensemble-unbiased estimator of $\langle M \rangle$ on
a single configuration [^4]. The output lies in $[-1, 1]$: values close to
$\pm 1$ denote uniformly polarised states along the field axis; values near
zero denote compensated, antiferromagnetic, or chiral textures (e.g.
skyrmion lattices).

### 4.2 Absolute magnetisation $|M|$ (Eq. 28)

$$
|M| = \frac{1}{N} \Big| \sum_{i \in \mathcal{D}} s_z(i) \Big|.
$$

Orientation-invariant order parameter. Useful when inverted domains or
spontaneous $\mathbb{Z}_2$ symmetry breaking may coexist within a single
configuration [^4]. Note that $\langle |M| \rangle \ne |\langle M \rangle|$
in general; in the ensemble estimator (Section 5) the absolute value is
taken before averaging.

### 4.3 Susceptibility $\chi$, spatial proxy (Eq. 30)

The thermodynamic definition

$$
\chi = \frac{1}{T} \big(\langle M^2 \rangle - \langle M \rangle^2\big) \cdot N
$$

requires an ensemble. With only one configuration, we use the static
fluctuation-dissipation sum rule

$$
\chi = \frac{1}{T} \sum_{\mathbf{r}} G(\mathbf{r}),
\qquad
G(\mathbf{r}) = \frac{1}{N} \sum_{i \in \mathcal{D}}
                s_z(i)\,s_z(i + \mathbf{r}) - \bar{s}_z^2,
$$

with $\bar{s}_z = M$. The connected correlation $G(\mathbf{r})$ is computed
via the Wiener-Khinchin theorem, which states that the autocorrelation of a
square-integrable function is the inverse Fourier transform of its power
spectrum:

$$
G(\mathbf{r}) = \mathcal{F}^{-1}\big\\{|\mathcal{F}\\{s_z - \bar{s}_z\\}|^2\big\\}.
$$

#### Why a radial cutoff is needed

If $G(\mathbf{r})$ is summed over the full grid, Parseval's identity
trivialises the result to

$$
\sum_{\mathbf{r}} G(\mathbf{r}) = N \cdot \big(\overline{s_z^2} - \bar{s}_z^2\big),
$$

which is just $N$ times the spatial variance: it carries no information
about the spatial range of correlations and saturates at the same value for
any rearrangement of the spins. The library therefore truncates the sum at
$|\mathbf{r}| \le r_{\text{cut}}$ with default $r_{\text{cut}} = r_d / 2$,
which preserves sensitivity to the actual correlation length while staying
in the regime $\xi \ll \ell \ll r_d$. The cutoff is exposed as a parameter
(`r_cutoff`) and can be swept to verify robustness.

Validity reference: Newman & Barkema [^9], Section 3.7; Goldenfeld [^17].
The spatial proxy coincides with the thermodynamic $\chi$ when the
configuration is translationally invariant and self-averaging.

### 4.4 Exchange energy density $E$ (Eq. 32)

$$
E = -\frac{1}{N} \sum_{\substack{\langle i, j\rangle \\\\ i, j \in \mathcal{D}}}
            s_z(i)\,s_z(j),
$$

with the sum running over nearest-neighbour pairs along rows and columns,
each pair counted once. Pairs crossing the disk boundary $\partial \mathcal{D}$
are excluded by requiring both endpoints inside $\mathcal{D}$. The exchange
coupling $J$ is set to unity.

A fully aligned ferromagnetic state inside the disk gives $E \approx -1$,
while a fully disordered state gives $E \approx 0$.

**What this captures and what it does not.** The scalar image carries only
the diagonal $s_z s_z$ component of the full Heisenberg exchange
$\mathbf{s}_i \cdot \mathbf{s}_j$. The transverse components $s_x, s_y$,
further-neighbour exchanges $J_3, J_4$, the antisymmetric DMI term, Zeeman
coupling, and the magnetocrystalline anisotropy are not recoverable from a
single $s_z$ map. In the inverse-cycle setting these are absorbed implicitly
through the parameter conditioning of the generative model.

References: Mohylna & Žukovič [^4], Eq. 1; Jha [^18], Eq. 18.

### 4.5 Specific heat $C_v$, subsystem proxy (Eqs. 34-35)

The thermodynamic definition

$$
C_v = \frac{1}{N T^2} \big(\langle \mathcal{H}^2 \rangle - \langle \mathcal{H} \rangle^2\big)
$$

again requires an ensemble. The library replaces the ensemble variance by the
variance of the local energy density across non-overlapping spatial blocks &mdash;
Binder's subsystem fluctuation method [^5][^4]:

$$
C_v = \frac{|\mathcal{B}|}{T^2}\big(\langle \varepsilon^2 \rangle_{\mathcal{B}}
                                  - \langle \varepsilon \rangle_{\mathcal{B}}^2\big),
$$

with

$$
\varepsilon_k = -\frac{1}{|\mathcal{B}_k \cap \mathcal{D}|}
                 \sum_{\substack{\langle i, j\rangle \\\\ i, j \in \mathcal{B}_k \cap \mathcal{D}}}
                 s_z(i)\,s_z(j),
$$

and $\langle \cdot \rangle_{\mathcal{B}}$ averaging over the $K$ valid blocks.

* $\mathcal{B}_k$: $k$-th non-overlapping square block of side $\ell$ pixels.
  The library uses $\ell = 5$, $|\mathcal{B}| = \ell^2 = 25$.
* Only blocks with $|\mathcal{B}_k \cap \mathcal{D}| = \ell^2$ (fully inside
  the disk) are retained; this yields approximately 50 valid blocks on the
  $40 \times 40$ grid with $r_d = 18.3$.
* The factor $|\mathcal{B}|$ rescales the block-wise variance of the
  intensive energy density to an estimate of the extensive energy variance.

**Validity window.** The proxy is valid when the correlation length
satisfies $\xi \ll \ell \ll r_d$. Away from phase transitions this holds
comfortably. Near criticality, where $\xi \to r_d$, the proxy
underestimates the peak of $C_v$ &mdash; the standard finite-size limitation of
the method [^5].

---

## 5. Physical metrics &mdash; regime (b): ensemble

The DDPM is stochastic and can produce $K$ independent samples
$\\{X^{(k)}\\}_{k=1}^{K_{\text{ens}}}$ conditioned on the same parameter
vector. With this ensemble in hand, $M$, $|M|$, $E$ are reported as ensemble
means of the per-sample spatial estimators, and $\chi$, $C_v$ come directly
from the thermodynamic FDT definitions on the ensemble variance &mdash; no spatial
proxy is involved.

### 5.1 Magnetisation $\langle M \rangle$ (Eq. 27)

$$
\langle M \rangle = \frac{1}{K_{\text{ens}}} \sum_{k=1}^{K_{\text{ens}}}
                    \frac{1}{N} \sum_{i \in \mathcal{D}} s_z^{(k)}(i).
$$

### 5.2 Absolute magnetisation $\langle |M| \rangle$ (Eq. 29)

$$
\langle |M| \rangle = \frac{1}{K_{\text{ens}}} \sum_{k=1}^{K_{\text{ens}}}
                      \Big|\frac{1}{N} \sum_{i \in \mathcal{D}} s_z^{(k)}(i)\Big|.
$$

The absolute value is taken **before** the ensemble average.

### 5.3 Susceptibility $\chi$ (Eq. 31)

Direct thermodynamic estimator on the ensemble:

$$
\chi = \frac{N}{T}\big(\langle M^2 \rangle_{\text{ens}}
                     - \langle M \rangle_{\text{ens}}^2\big),
$$

with

$$
\langle M^p \rangle_{\text{ens}}
= \frac{1}{K_{\text{ens}}} \sum_{k=1}^{K_{\text{ens}}} M_k^p,
\qquad
M_k = \frac{1}{N} \sum_{i \in \mathcal{D}} s_z^{(k)}(i).
$$

Returns NaN if $T = 0$ or $K_{\text{ens}} < 2$.

### 5.4 Exchange energy density $\langle E \rangle$ (Eq. 33)

$$
\langle E \rangle = \frac{1}{K_{\text{ens}}} \sum_{k=1}^{K_{\text{ens}}}
                    \bigg(-\frac{1}{N}
                    \sum_{\substack{\langle i, j\rangle \\\\ i, j \in \mathcal{D}}}
                    s_z^{(k)}(i)\,s_z^{(k)}(j)\bigg).
$$

### 5.5 Specific heat $C_v$ (Eq. 36)

Direct thermodynamic estimator on the ensemble:

$$
C_v = \frac{N}{T^2}\big(\langle E^2 \rangle_{\text{ens}}
                     - \langle E \rangle_{\text{ens}}^2\big),
$$

with $E_k$ the per-sample exchange energy density. Returns NaN if $T = 0$ or
$K_{\text{ens}} < 2$.

---

## 6. Why two regimes

The two regimes do not measure the same quantity in different ways &mdash; they
make different statistical assumptions:

* The **single-configuration estimator** in regime (a) treats the spatial
  mean as a proxy for the ensemble mean. It is valid only when the
  configuration is large enough that spatial averaging mimics ensemble
  averaging (the self-averaging hypothesis, [^17] Sec. 7).
* The **ensemble estimator** in regime (b) is the textbook
  fluctuation-dissipation definition, and is the reference whenever an
  ensemble of independent realisations is available.

The article exploits this distinction as an internal calibration mechanism:
on the generated set both estimators can be computed in parallel, and their
agreement (or disagreement) serves as empirical evidence supporting (or
challenging) the use of the spatial proxy on the simulated set where only
the single configuration is available [^4].

For this reason the library exposes both as separate functions with explicit
`_orig` and `_gen` suffixes, rather than dispatching automatically based on
input shape.

---

## 7. Limitations and validity windows

* **Resize attenuation.** Bilinear interpolation from $(39, 39)$ to
  $(40, 40)$ attenuates extreme spin values on the ~1-px ring at the disk
  perimeter. The effect is bounded but systematic and affects regime (a)
  more than regime (b) (the generator output is native at $40 \times 40$).
* **Scalar-image limitation of $E$.** Only the $s_z s_z$ component of the
  full Heisenberg exchange is recoverable. Further-neighbour exchanges, DMI,
  Zeeman, and anisotropy are absorbed implicitly through the parameter
  conditioning of the generative model and cannot be reconstructed from the
  image alone.
* **Finite-size effects in $\chi$ (regime a).** When the correlation length
  approaches $r_d$ (near a phase transition), the spatial proxy
  underestimates the true thermodynamic $\chi$. The radial cutoff
  $r_{\text{cut}} = r_d / 2$ mitigates this but does not eliminate it.
* **Finite-size effects in $C_v$ (regime a).** The Binder subsystem proxy is
  valid only in the window $\xi \ll \ell \ll r_d$. Near criticality this
  window collapses and the proxy underestimates the peak of $C_v$.
* **Sample size in regime (b).** The ensemble variance is a noisy estimator
  for small $K_{\text{ens}}$. The library returns NaN for
  $K_{\text{ens}} < 2$. Practical convergence of $\chi$ and $C_v$ typically
  requires $K_{\text{ens}} \gtrsim 32$.
* **Temperature normalisation.** $\chi$ and $C_v$ require $T > 0$; both
  return NaN for $T = 0$. The library does not check the upper bound of $T$,
  but values far above the exchange scale ($T \gg J / k_B$) will yield
  uninformative results because the system is paramagnetic.

---

## 8. References

[^4]: Mohylna, M. and Žukovič, M. (2022). Skyrmion lattice phases in a
    two-dimensional easy-axis Heisenberg ferromagnet with Dzyaloshinskii-Moriya
    interaction. Equations 1-4. The library follows the same convention for
    $M$, $|M|$, $\chi$, $C_v$ in both regimes.
[^5]: Binder, K. (1981). Finite size scaling analysis of Ising model block
    distribution functions. *Zeitschrift für Physik B*, 43(2), 119-140. The
    subsystem fluctuation method for $C_v$.
[^9]: Newman, M. E. J. and Barkema, G. T. (1999). *Monte Carlo Methods in
    Statistical Physics*. Oxford University Press. Section 3.7 derives the
    FDT sum rule used in the spatial proxy of $\chi$.
[^11]: Article Eq. 22. Variance of the per-pixel squared error.
[^16]: Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004).
    Image quality assessment: from error visibility to structural similarity.
    *IEEE Transactions on Image Processing*, 13(4), 600-612.
[^17]: Goldenfeld, N. (1992). *Lectures on Phase Transitions and the
    Renormalization Group*. Addison-Wesley. Self-averaging and the validity
    of spatial proxies.
[^18]: Jha, P. K. (2026). *Image-based analysis of magnetic textures*. Eq. 18,
    cited in the article for the single-configuration estimator of $E$.

Additional references &mdash; cited throughout the article and underlying the
broader framework around this library:

* Landau, D. and Binder, K. (2014). *A Guide to Monte Carlo Simulations in
  Statistical Physics*. Cambridge University Press, Ch. 4. The reference text
  for $|M|$, $\chi$, and $C_v$ as order parameters.
* Eriksson, O., Bergman, A., Bergqvist, L., and Hellsvik, J. (2017).
  *Atomistic Spin Dynamics: Foundations and Applications*. Oxford University
  Press. Foundations of the extended Heisenberg Hamiltonian.
* Evans, R. F. L. et al. (2014). Atomistic spin model simulations of magnetic
  nanomaterials. *Journal of Physics: Condensed Matter*, 26(10), 103202.