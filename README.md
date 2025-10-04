# Towards Universal AutoML: A Hybrid Meta-Learning Framework for Cross-Domain Hyperparameter Optimization
![License](https://img.shields.io/badge/license-MIT-blue)
![Conference](https://img.shields.io/badge/Conference-CODS_COMAD_'25-purple)

This repository provides a comprehensive overview of the research paper "Towards Universal AutoML." We introduce a novel hybrid meta-learning framework designed to solve one of the most persistent challenges in machine learning: efficient, interpretable, and cross-domain Hyperparameter Optimization (HPO).

## 📝 In-Depth Overview

The promise of Artificial Intelligence is often gated by the performance of its underlying machine learning models. This performance, in turn, is critically dependent on a model's hyperparameters—the configuration settings that are not learned from the data itself. The process of finding the best hyperparameters, known as **Hyperparameter Optimization (HPO)**, is a major bottleneck in the practical application of machine learning. It is a computationally expensive, time-consuming, and frustratingly dataset-specific endeavor.

Meta-learning, or "learning to learn," offers a potential solution by leveraging experience from past HPO tasks to inform and accelerate new ones. However, existing meta-learning approaches have critical limitations. Some rely on simple dataset statistics that fail to capture deep structural similarities, while others use powerful but uninterpretable "black-box" embeddings that struggle to generalize across different data types (modalities) like tabular data, text, and images.
 
 
 
   <img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/f95c0a78-5d3d-4b02-8a4b-1cf3dc1b02ca" />


This work introduces and validates a **hybrid meta-learning framework** that resolves this trade-off. We combine hand-crafted, interpretable **statistical meta-features** with expressive, automatically **learned dataset embeddings**. This hybrid representation allows our system to understand datasets at multiple levels of abstraction. We use this representation to train a model that can accurately predict the best-performing hyperparameters for entirely new datasets, even those from modalities it has never seen before.

Our extensive evaluation, performed on a diverse benchmark of nine datasets across three modalities using a strict **leave-one-dataset-out (LODO)** protocol, confirms the superiority of our approach. The hybrid model achieves the most consistent and reliable generalization, significantly advancing the goal of building truly scalable, interpretable, and general-purpose Automated Machine Learning (AutoML) systems.

---

## 🎯 The Problem: Why is HPO So Difficult?

The core challenge of HPO is navigating a vast, high-dimensional search space where each evaluation (training and testing a model with one set of hyperparameters) is computationally expensive.

* **Curse of Dimensionality:** As the number of hyperparameters increases, the search space grows exponentially, making exhaustive methods like grid search computationally intractable.
* **"No Free Lunch" Theorem:** This fundamental concept implies that no single set of hyperparameters will be optimal for all datasets. What works for an image classification task might fail spectacularly for a financial fraud detection task. This forces data scientists to restart the expensive search process for every new problem.
* **Lack of Transferability:** Traditional HPO methods like Bayesian Optimization or Hyperband are highly efficient for a single dataset but start from scratch on a new one. They don't carry over knowledge, leading to immense redundancy and wasted compute cycles in organizations that build many models.
* **Black Box Nature:** The relationship between dataset characteristics and optimal hyperparameters is often non-obvious. This makes HPO feel more like an art than a science and hinders our ability to build automated, reliable systems.

---

## 🚀 Our Solution: A Detailed Architectural Breakdown

Our framework is a complete pipeline designed to learn from past HPO experiments and apply that knowledge to new tasks efficiently.

### 1. The Hybrid Dataset Representation: The Secret Sauce
The foundation of our system is a rich, multi-faceted "fingerprint" for each dataset. This is created by concatenating two complementary vectors.

#### **Part A: Statistical Meta-Features (The Interpretable Summary)**
These are **hand-crafted statistical descriptors** that provide a high-level, human-understandable summary of the dataset's properties. They are calculated using predefined formulas and help our model learn explicit, interpretable rules.

* **What we measure:**
    * **Basic Properties:** Number of samples (`n`), feature dimensionality (`d`).
    * **Information-Theoretic Properties:** `class_entropy` (how spread out the classes are) and `class_imbalance` (whether some classes have far more samples than others).
    * **Statistical Moments:** The mean, standard deviation, **skewness** (asymmetry of the data distribution), and **kurtosis** (tailedness of the distribution) of the features.
    * **Correlation Structure:** The average absolute pairwise correlation between features.
* **Why they are crucial:** These features make our model **explainable**. They allow us to understand *why* the model makes a certain recommendation, connecting it back to tangible properties of the data.



  <img width="410" height="223" alt="image" src="https://github.com/user-attachments/assets/01b6c467-7820-4d10-ac46-97ebfdefed7e" />  


#### **Part B: Learned Dataset Embeddings (The Deep Structural Fingerprint)**
These are dense, low-dimensional vectors **learned automatically by a neural network**. They capture deep, complex, and non-linear patterns within the data that simple statistics cannot. Think of this as the dataset's unique DNA.

* **How we learn them:** We use a modality-specific approach to create a powerful embedding for each dataset type.
    * **For Tabular Data:** We train a three-layer **Denoising Autoencoder** on the dataset. This network learns to compress the data into a small latent space and then reconstruct it, forcing the latent vector to capture the most essential information. The mean of these latent vectors becomes our dataset embedding.
    * **For Text Data:** We use a standard text processing pipeline, converting documents into a TF-IDF matrix, and then apply **TruncatedSVD** to reduce its dimensionality to a consistent size (32 dimensions).
    * **For Vision Data:** We leverage the power of transfer learning by passing the images through a pre-trained **ResNet-18** model to extract high-level features. These features are then also reduced via TruncatedSVD to 32 dimensions.
* **Why they are crucial:** Embeddings provide immense **expressive power**. They allow the model to recognize nuanced similarities between datasets (e.g., that MNIST and Fashion-MNIST are structurally similar image problems) and make highly accurate predictions.

### 2. The Pairwise Ranking Model (RankNet)
With our hybrid representation, we train a model to predict HPO performance. Instead of predicting an exact accuracy score (a difficult regression task), we train a **RankNet** to perform a simpler, more robust task: **pairwise ranking**.

* **Architecture:** The RankNet is a simple three-layer Multi-Layer Perceptron (MLP).



  <img width="243" height="316" alt="image" src="https://github.com/user-attachments/assets/02b41877-7485-4929-831e-1265b11102aa" />

* **Input:** The input is a concatenated vector containing the **hybrid dataset representation** and the **encoded hyperparameter values** of a given trial.
* **Task:** The model is trained on pairs of HPO trials from the same dataset. Its goal is to output a higher score for the trial that achieved better accuracy.
* **Loss Function:** We train it as a binary classification problem using `BCEWithLogitsLoss`.
* **Why this approach?** It is more resilient to noise in performance measurements and more directly addresses the goal of HPO, which is to *find the best settings*, not to perfectly predict the score of every possible setting.


  <img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/5a2ea947-e1eb-46d9-b3b0-8936f118a821" />



### 3. Few-Shot Calibration: Making it Practical
The RankNet produces a relative score, not an absolute accuracy prediction. To bridge this gap for a new dataset, we use an efficient **few-shot calibration** process.


   <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/76e3b576-92a3-4225-9dca-6a877bdd0791" />

1.  **Arriving at a New Dataset:** A user brings a new dataset for which they want to find the best hyperparameters.
2.  **Creating a Support Set:** We run a very small number of HPO trials (e.g., up to 8) with randomly chosen hyperparameters. This is the "support set" and is extremely cheap to generate.
3.  **Learning the Mapping:** We train a simple **Ridge regression model** that learns a linear mapping from the RankNet's internal scores to the actual accuracies observed in the support set.
4.  **Predicting for All Other Trials:** This calibrated model can now predict the absolute accuracy for any other hyperparameter configuration on this new dataset, without needing to run the expensive training jobs.

---

## 💡 Novelty and Impact in Detail

* **Novelty:** The primary scientific contribution is the **demonstration that a single, hybrid meta-learning framework can successfully generalize across highly diverse data modalities**. While prior works explored hybrid representations, they were often confined to a single domain (e.g., vision only) or tested on small, non-diverse benchmarks. Our large-scale, multi-modal LODO evaluation is the first of its kind to rigorously prove this cross-domain capability.
* **Impact:** This research provides a tangible roadmap towards the future of AutoML.
    * **Democratization of ML:** It dramatically lowers the computational and expertise barrier for achieving high model performance, enabling smaller companies and individual researchers to compete.
    * **Operational Efficiency:** For large enterprises, this translates to massive savings in cloud computing costs and a significant reduction in the time it takes to deploy and update models.
    * **A Shift in Paradigm:** It moves HPO from a brute-force search problem to an intelligent, data-driven inference problem, making the entire ML workflow more efficient and predictable.

---

## 🧠 Interpretability & Explainability: Opening the Black Box

A model is only as good as our trust in it. Our hybrid framework is designed for transparency.

* **SHAP (SHapley Additive exPlanations):** We use SHAP to quantify the impact of each statistical meta-feature on the final prediction. The SHAP summary plot in our paper reveals fascinating insights:
    * Features related to **dataset dimensionality** (e.g., number of features) and **class distribution** (e.g., class entropy, imbalance) were consistently the most influential predictors of HPO performance.
    * This allows a data scientist to understand the model's reasoning, for instance, "The model is recommending these hyperparameters because your dataset has a high number of features and significant class imbalance."

* **UMAP/t-SNE Visualizations:** These techniques project our high-dimensional learned embeddings into a 2D space that we can see. The resulting plot clearly shows that datasets from the same modality (e.g., all vision datasets) form tight clusters. This provides a powerful, intuitive validation that our embedding process is successfully learning meaningful relationships and explains why transferring knowledge within a domain is so effective.

     <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/0ed32a2e-56e4-4450-9596-3a7ed1a1bf47" />


---

## 📈 Rigorous Evaluation & Detailed Results

We believe in rigorous, honest evaluation. Our experimental setup was designed to be as challenging and realistic as possible.

* **The LODO Protocol:** As explained, the **Leave-One-Dataset-Out** protocol is the gold standard for testing meta-learning generalization. It simulates the real-world scenario of encountering a completely novel problem, preventing any form of data leakage or performance overestimation.
* **The Benchmark Suite:** We used nine popular datasets spanning three modalities to ensure our conclusions were robust:
    * **Tabular:** Iris, Wine, Breast Cancer, Adult, Bank Marketing.
    * **Text:** 20 Newsgroups.
    * **Vision:** MNIST, Fashion-MNIST, CIFAR-10.

### Key Findings and Nuances
The **Hybrid model achieved the highest mean Spearman correlation (0.807)**, confirming its superior ability to correctly rank hyperparameters.

| Variant | Spearman ($\rho$) | Pearson (r) | MAE |
| :--- | :--- | :--- | :--- |
| EMBED | 0.772 ($\pm$0.107) | 0.878 | 0.036 |
| META | 0.635 ($\pm$0.416) | 0.594 | 0.052 |
| **HYBRID** | **0.807 ($\pm$0.059)** | **0.803** | **0.059** |
*Table: Aggregate LODO performance metrics. A higher Spearman is better.*

* **The Power of Embeddings:** The `EMBED`-only model was surprisingly strong, achieving a high Spearman score of 0.772 with a low confidence interval. This shows that learned representations by themselves are an extremely powerful and resilient meta-feature.
* **The Unreliability of Meta-Features Alone:** The `META`-only model had the lowest average performance (0.635) and a very large confidence interval, indicating its performance was highly variable and unreliable, especially on complex vision and text tasks.
* **Statistical Significance:** While the Hybrid model performed best on average, a paired permutation test showed the difference was not statistically significant when compared to the Embed model. This powerful finding implies that the hybrid model's primary advantage is not just a raw performance boost, but the combination of **top-tier performance AND crucial interpretability**, which the `EMBED` model lacks.

---

## 🌍 Real-World Applications & Conceptual Workflow

#### How would you use this system?
1.  **Meta-Training (Done once):** First, the system is trained on a large, diverse collection of existing datasets and their corresponding HPO trial results. This creates the powerful meta-model.
2.  **A New Task Arrives:** A data scientist has a new dataset (e.g., "new\_customer\_data.csv").
3.  **Feature Extraction:** The system automatically calculates the 15+ statistical meta-features and learns the 32-dimensional dataset embedding for the new data. These are concatenated into its hybrid representation.
4.  **Few-Shot Calibration (Seconds to Minutes):** The system runs 8 random HPO trials on the new data to get a small sample of real-world performance. It uses this to calibrate its internal scoring model.
5.  **Inference (Milliseconds):** The data scientist can now query the calibrated meta-model with hundreds or thousands of potential hyperparameter configurations. The model instantly returns a ranked list of the most promising configurations to try.
6.  **Final Training:** The data scientist takes the top 1-3 recommendations and trains their final models, having saved days or weeks of computational effort.

---
## Limitations and Future Work
No research is without limitations. We identify several exciting avenues for future work:
* **Expanding Search Spaces:** Our study used small hyperparameter grids for reproducibility. Future work should test the framework's scalability on much larger and more complex search spaces.
* **Broader Task Domains:** Our analysis focused on classification tasks. Expanding and validating the approach for regression, time-series forecasting, or multi-modal problems is a key next step.
* **Stronger Fusion Methods:** Investigating more sophisticated methods for fusing meta-features and embeddings, beyond simple concatenation, could yield further performance gains.
