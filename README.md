# Domain Adaptation in Computer Vision

**Evaluating Feature Alignment Methods for Hand-Written Digit Classification (MNIST → USPS)** **Author:** Aleksandar Ginev  
**Project Type:** Technical Report / Machine Learning Research  

---

## Abstract

This research investigates the phenomenon of *Domain Shift* within computer vision architectures. When convolutional models or standard classifiers are trained on pristine, synthetic, or perfectly curated imagery, their performance severely degrades when deployed on noisy, real-world target domains. 

Using the classic MNIST (Source) and USPS (Target) datasets, we analyze this performance drop and implement standard Unsupervised Domain Adaptation (UDA) techniques—specifically Correlation Alignment (CORAL)—to mathematically align the latent feature spaces. Our results demonstrate that optimizing the covariance matrices of source and target domains significantly recovers classification accuracy without requiring target labels during training.

---

## 1. Introduction & Problem Formulation

In modern computer vision systems, maintaining high generalization accuracy across shifting visual environments is a persistent challenge. A model trained on clean, digital images often fails when exposed to real-world factors like sensor noise, lighting variations, or changes in handwriting style. 

The objective of this project is to model this "Domain Drift" explicitly by training a classifier on the MNIST dataset and evaluating it on the USPS dataset. We formulate this as an Unsupervised Domain Adaptation task, where the goal is to map both domains into a shared, invariant feature space where a single classifier can succeed.

---

## 2. Data Acquisition & Vetting

Our pipeline utilizes two primary, structurally distinct benchmark datasets containing handwritten digits (0–9):

- **MNIST (Source Domain):** 70,000 clean, centered, 28x28 pixel grayscale images.  
- **USPS (Target Domain):** 9,298 uncentered, blurred, and lower-contrast 16x16 pixel grayscale images sampled from actual US Postal Service mail envelopes.  

The vetting and ingestion process involved:
- **Spatial Normalization:** Resampling and resizing the USPS matrices to 28x28 pixels to establish a structural 1:1 input vector compatibility.
- **Min-Max Scaling:** Normalizing pixel intensity values to the range [0, 1] to stabilize gradient descent during optimization.

---

## 3. Mathematical Foundations

To identify and correct the spatial drift between domains, we utilize several key mathematical tools:

- **Pixel Vector Representation:** Flattening 28x28 matrices into feature vectors $x \in \mathbb{R}^{784}$.
- **Covariance Alignment (CORAL):** Computing the second-order statistics (covariance matrices $\Sigma_S$ and $\Sigma_T$) of both domains to find a linear transformation matrix $A$ that minimizes the distance between them.
- **Maximum Mean Discrepancy (MMD):** Measuring the distance between probability distributions in a Hilbert space.
- **Softmax Cross-Entropy:** Modeling multi-class categorical probabilities across the 10 digit classes.

---

## 4. Data Pre-processing & Feature Engineering

Because raw pixels contain high levels of redundancy, feature processing is vital. Our workflow executes three critical actions:

- **Dimensionality Matching:** Bilinear interpolation of target images to match feature vector dimensions.
- **Mean Centering:** Subtracting the empirical mean of the datasets to align the coordinates around a global origin.
- **Covariance Regularization:** Adding a small identity matrix perturbation ($\lambda I$) to guarantee that the source covariance matrix is invertible during alignment.

---

## 5. Exploratory Data Analysis (EDA)

EDA was executed to validate our core hypotheses regarding Domain Shift:

- **Visual Comparison:** Visualizing raw pixel grids side-by-side to highlight differences in contrast, thickness, and edge noise.
- **Dimensionality Reduction (PCA & t-SNE):** Projecting the 784-dimensional pixel data into 2D space. The resulting scatter plots reveal a distinct spatial separation between the MNIST cluster and the USPS cluster, visually proving the presence of *Domain Drift*.

---

## 6. Modeling & Hypothesis Testing

We establish a baseline using standard classifiers (Logistic Regression / Shallow Neural Network) trained exclusively on the Source domain.

- **Target Variable:** Categorical Digit Label (0–9)  
- **Source Baseline Accuracy:** ~98% when tested on MNIST validation data.
- **Cross-Domain Drop:** A severe drop in accuracy (e.g., down to ~65%) when the unadapted baseline model is evaluated directly on the raw USPS dataset.
- **Experiment Tracking:** All performance drops, hyperparameters, and accuracy shifts are recorded dynamically via **MLflow**.

---

## 7. Adaptation Strategy: Latent Space Alignment

We implement the Domain Adaptation mechanism directly on the extracted features:

- **Feature Whitening:** Transforming the source features to have an identity covariance matrix.
- **Coloring Transformation:** Re-coloring the whitened source features with the precise covariance structure of the target domain ($\Sigma_T$).

This mathematical transformation allows our source classifier to interpret target features smoothly, without modifying the underlying classification layer weights.

---

## 8. Results & Interpretation

The implementation of the alignment transformation yields quantifiable performance recoveries:

- **Accuracy Metrics:** Post-adaptation testing on the USPS dataset shows a significant boost in classification accuracy, proving the validity of the alignment matrix.
- **Confusion Matrix Analysis:** Pre-adaptation models frequently confused digits with similar structural profiles (e.g., 3 vs. 8, or 4 vs. 9) due to target blur. The post-adaptation confusion matrix displays a highly stabilized diagonal layout, indicating clear error reduction.
- **System Footprint:** The algorithm achieves this optimization linearly, preserving computational efficiency without requiring heavy deep-learning fine-tuning.

---

## 9. Conclusion & References

This project confirms that Domain Drift in computer vision is a highly predictable, mathematically correctable phenomenon. By utilizing simple, second-order feature statistics, we successfully bridge the gap between two completely different historical digit scanning systems.

Future extensions of this report include upgrading the linear feature alignment to deep adversarial adaptation (DANN) using deep convolutional layers.

---

## References

- LeCun, Y., Cortes, C., & Burges, C. (1998). *The MNIST database of handwritten digits.*
- Hull, J. J. (1994). *A database for handwritten text recognition research.* (USPS Dataset)
- Sun, B., Feng, J., & Saenko, K. (2016). *Return of Frustratingly Easy Domain Adaptation.* (CORAL Algorithm)
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.*
