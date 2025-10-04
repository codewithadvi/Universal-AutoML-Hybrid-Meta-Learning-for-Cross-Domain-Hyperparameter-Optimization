# Towards Universal AutoML: A Hybrid Meta-Learning Framework for Cross-Domain Hyperparameter Optimization

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Conference](https://img.shields.io/badge/Conference-CODS_COMAD_'25-purple)

This repository contains the conceptual framework and findings of the paper "Towards Universal AutoML," which introduces a novel hybrid meta-learning approach for efficient, interpretable, and cross-domain Hyperparameter Optimization (HPO).

## 📝 Overview

Building high-performance machine learning models is often bottlenecked by Hyperparameter Optimization (HPO), an expensive and dataset-specific process. Current meta-learning techniques attempt to transfer knowledge from past experiments but often fail to generalize across different data modalities (e.g., from text to vision).

Our work proposes a **hybrid meta-learning framework** that combines the strengths of interpretable statistical meta-features and expressive learned dataset embeddings. This unified approach accurately predicts model performance and guides HPO across diverse domains, including tabular, text, and vision datasets. Through a rigorous leave-one-dataset-out evaluation, we demonstrate that our hybrid model achieves superior generalization, advancing the goal of creating scalable, interpretable, and general-purpose AutoML systems.

---

## 🎯 The Problem: The HPO Bottleneck

Hyperparameter tuning is a fundamental yet challenging problem in machine learning. State-of-the-art algorithms are highly sensitive to their hyperparameter settings, and finding the optimal configuration is:
* **Computationally Expensive:** Traditional methods like grid search and random search are exhaustive and costly.
* **Dataset-Dependent:** Optimal hyperparameters for one dataset rarely transfer well to another, forcing practitioners to restart the expensive search process for each new task.
* **A Scalability Barrier:** In industrial settings (e.g., finance, healthcare), the cost of re-tuning models on new data poses a massive obstacle to efficiency and rapid deployment.

---

## 🚀 Our Solution: A Hybrid Meta-Learning Framework

We tackle this challenge with a three-part pipeline that learns to recommend optimal hyperparameters for new, unseen datasets with minimal effort.

### 1. Hybrid Dataset Representation
The core of our framework is a novel way to represent any given dataset. We create a rich feature vector by combining two complementary sources of information:

* **Statistical Meta-Features:** These are **hand-crafted, interpretable descriptors** that summarize a dataset's characteristics. They include properties like:
    * Sample size and feature dimensionality.
    * Information-theoretic quantities like class entropy and imbalance.
    * Statistical moments like mean, skewness, and kurtosis of features.
    * *Why?* These features provide a human-understandable summary, making the model's reasoning transparent.

* **Learned Dataset Embeddings:** These are **dense vector representations** learned automatically by a neural network to capture deep, structural patterns in the data. We use different encoders depending on the modality:
    * **Tabular:** A Denoising Autoencoder learns a compressed representation.
    * **Text & Vision:** Pre-trained models (TruncatedSVD and ResNet-18) are used to generate powerful feature embeddings.
    * *Why?* Embeddings capture complex similarities between datasets that simple statistics miss, providing powerful predictive signals.

By concatenating these two representations, we create a hybrid vector that balances **interpretability with expressive capacity**.

### 2. Pairwise Ranking Model (RankNet)
Instead of predicting a noisy, absolute accuracy score, our model learns a more robust task: **ranking**. We use a simple MLP-based RankNet that, given two different hyperparameter configurations, predicts which one is more likely to perform better. This approach is more resilient and aligns directly with the goal of HPO: finding the *best* settings.

### 3. Few-Shot Calibration
To make the model practical, we use a **few-shot calibration** technique. When encountering a new, unseen dataset, we run a very small number of random trials (e.g., 2-8) to create a "support set". A simple Ridge regression model then learns to map the RankNet's relative scores to the true accuracy values observed in the support set. This calibrated model can then accurately predict the performance of all other hyperparameter configurations at a fraction of the cost of running them.

---

## 💡 Novelty and Impact

* **Novelty:** This is one of the first works to propose and **rigorously evaluate a single, hybrid meta-learning framework across diverse modalities** (tabular, text, and vision) at scale. It bridges the gap left by prior research, which was often limited to a single domain or smaller benchmarks.
* **Impact:** This research provides a scalable, interpretable, and economical path towards **zero-shot and few-shot HPO**. It significantly lowers the barrier to building high-quality ML models, empowering developers to achieve state-of-the-art performance without extensive, costly tuning. This is a critical step towards the next generation of general-purpose AutoML systems.

---

## 🧠 Interpretability & Explainability

A key feature of our framework is its ability to provide insights into *why* certain hyperparameters work well for a given dataset. We use two main techniques:

* **SHAP (SHapley Additive exPlanations):** We use SHAP to understand the impact of our statistical meta-features on the model's predictions. The SHAP summary plot reveals that dataset characteristics like **dimensionality (`n_features`)** and **class distribution (`class_entropy`, `imbalance`)** are the most influential predictors of HPO performance. This creates a direct, explainable link between a dataset's properties and the tuning results.

* **UMAP/t-SNE Visualizations:** By projecting the learned dataset embeddings into 2D space, we can visually inspect the relationships between datasets. The visualizations confirm that datasets from the same modality (e.g., vision) cluster together, providing an intuitive reason for why knowledge transfer is successful within those domains.

---

## 📈 Rigorous Evaluation & Results

We validated our framework using a strict **Leave-One-Dataset-Out (LODO)** protocol, which ensures the model is always tested on a completely unseen dataset, providing a true measure of its generalization ability. Our evaluation spanned nine benchmark datasets across three modalities.

### Key Findings
The hybrid approach consistently delivered the most reliable generalization across all domains. The `EMBED` (embeddings only) and `META` (meta-features only) baselines were competitive, but the `HYBRID` model achieved the best overall performance in ranking hyperparameters correctly.

| Variant | Spearman ($\rho$) | Pearson (r) | MAE |
| :--- | :--- | :--- | :--- |
| EMBED | 0.772 ($\pm$0.107) | 0.878 | 0.036 |
| META | 0.635 ($\pm$0.416) | 0.594 | 0.052 |
| **HYBRID** | **0.807 ($\pm$0.059)** | **0.803** | **0.059** |
*Table: Aggregate LODO performance metrics. Spearman correlation is the primary metric for ranking quality.*

These results confirm our central hypothesis: combining statistical meta-features with learned embeddings creates a single, robust representation that generalizes effectively across domains.

---

## 🌍 Real-World Applications

This framework is designed for any scenario where ML models are frequently built on new datasets, saving significant time and computational resources.

* **Enterprise AI Platforms:** Cloud providers and MLOps platforms can integrate this system to offer "intelligent defaults" or a "zero-shot tuning" option, drastically accelerating customer workflows.
* **Financial Services:** A bank developing fraud or credit risk models for different customer segments can use this to instantly tune models for new segments without a full HPO cycle.
* **Healthcare:** Research institutions analyzing diverse medical datasets (e.g., patient records, medical images, genomic data) can rapidly optimize classifiers for new diagnostic tasks.
* **E-commerce:** Retail companies can efficiently tune personalized recommendation models for thousands of different product categories or user groups.
