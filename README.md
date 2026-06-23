# Domain Adaptation in Computer Vision

**Feature Alignment for Handwritten Digit Classification (MNIST → USPS)**

**Author:** Aleksandar Ginev
**Project Type:** Technical Report / Machine Learning

---

## What this project is about

Most ML courses teach train/test splits as if that's the whole story, but that only works if your train and test data come from the same distribution. In real life that's often not true. This project looks at one concrete case of that problem - called domain shift - using two handwritten digit datasets: MNIST (clean, scanned, centered) and USPS (real mail envelopes, lower resolution, much messier handwriting).

The setup is Unsupervised Domain Adaptation: I get full labels for MNIST, but none for USPS during adaptation. The goal is to train on MNIST and still do reasonably well on USPS, without ever looking at a USPS label until the final evaluation.

The main method I use is **CORAL** (Correlation Alignment, Sun & Saenko, 2016), which works by matching the covariance structure of the two domains instead of matching individual images.

## Data

- **MNIST (source):** 70,000 grayscale images, 28x28, clean and centered.
- **USPS (target):** 9,298 grayscale images, natively 16x16, scanned from real mail, noisier and less uniform handwriting.

Since CORAL needs both domains to have the same number of features, USPS images are upscaled from 16x16 to 28x28 with bilinear interpolation before anything else happens. Both datasets are then scaled to [0, 1].

(Note: OpenML flags the USPS version used here as "inactive" due to known issues with that release. I checked the images and labels after loading and they look correct, but it's worth mentioning in case results differ slightly from other USPS versions.)

## Method

1. **Baseline** - train a plain logistic regression on MNIST only, test it both on held-out MNIST and on raw, unadapted USPS. This shows how bad the problem actually is.
2. **Standardization-only baseline** - a much simpler fix: just z-score each domain on its own (`StandardScaler`), no covariance matching. This is here mainly as a sanity check on CORAL - if a basic rescale gets close to CORAL's result, CORAL isn't adding much.
3. **CORAL** - whiten the source features (remove their own covariance structure), then re-color them with the target's covariance structure. Train logistic regression on these adapted features, test on raw USPS.
4. **Sanity check** - directly verify that CORAL actually pulled the source covariance closer to the target covariance, using the Frobenius norm distance between them, before assuming the accuracy numbers mean anything.

All runs are logged with MLflow (classifier, params, accuracy per run) so the numbers below can be reproduced or compared against future changes.

## Results

| Setup | Target (USPS) accuracy |
|---|---|
| MNIST → MNIST (in-domain, for reference) | 92.11% |
| MNIST → USPS, no adaptation | 7.70% |
| MNIST → USPS, standardization only | 36.07% |
| MNIST → USPS, CORAL | **53.15%** |

CORAL recovers **+45.45 points** over doing nothing, and **+17.08 points** over the simple standardization baseline - so the covariance-matching part of CORAL is doing real work here, not just fixing scale differences.

The sanity check backs this up directly: the Frobenius distance between the source and target covariance matrices drops from **9.54 to about 0.0003** after CORAL, more than a 99.99% reduction. That's CORAL doing exactly what the math says it should - it isn't a coincidence that accuracy went up.

Per-digit performance (from the classification report) is uneven. Digits 0, 1, and 2 end up reasonably well classified (F1 around 0.67-0.74), but 5 and 9 stay weak even after adaptation (F1 around 0.08-0.12). My guess is that CORAL only matches second-order statistics (covariance), not finer shape detail, so digits that get distorted the most by the 16x16 → 28x28 upscaling are still hard to recognize even once the overall feature distributions line up.

## What I'd try next

A non-linear or deep version of this idea (e.g. Deep CORAL, or a fully adversarial approach like DANN) would probably close more of the remaining gap, especially for the digits that stay hard - but that's beyond what's covered in this notebook.

## Repo structure

- `domain_adaptation_in_computer_vision.ipynb` - the full notebook: data loading, EDA, baseline, standardization comparison, CORAL implementation, and evaluation.

## References

- Sun, B., & Saenko, K. (2016). *Deep CORAL: Correlation Alignment for Deep Domain Adaptation.* CVPR Workshops.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE, 86(11), 2278-2324.
- Hull, J. J. (1994). *A database for handwritten text recognition research.* IEEE TPAMI, 16(5), 550-554.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
