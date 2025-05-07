# Human Value Detection in Argument Mining

This project addresses the challenge of **multi-label classification** to detect **human values** expressed in textual arguments using state-of-the-art NLP models.

## 🧠 Project Overview

Human values such as *tolerance*, *objectivity*, and *humility* shape human behavior and social understanding. Recognizing these values in arguments has real-world relevance in social sciences, education, and media analysis.

We implemented and compared three models:
- **SVM** (baseline traditional model)
- **BERT** (transformer-based model)
- **XLNet** (advanced autoregressive transformer)

## 🗃️ Dataset

- Provided by the [Touché Human Value Detection Challenge 2023](https://touche.webis.de/semeval23/touche23-web/)
- Each sample includes: `Premise`, `Stance`, and `Conclusion`
- Labels: 20 human value categories (based on Schwartz's theory of basic values)
- Dataset split:
  - Train: 5,393 samples
  - Validation: 1,896 samples
  - Test: 1,578 samples

## ⚙️ Models & Approach

### 1. SVM
- Preprocessing:
  - Lowercasing, punctuation removal, stopword filtering
  - TF-IDF vectorization
  - Feature scaling with `StandardScaler`
  - Dimensionality reduction using `TruncatedSVD` (95% variance, 710 components)
- Multi-label classification with `MultiOutputClassifier`
- Hyperparameter tuning via **Ray Tune**

### 2. BERT & XLNet
- Minimal preprocessing; leveraged tokenization
- Implemented with `SimpleTransformers`
- Used W&B **Sweeps** for hyperparameter tuning
- Evaluated both base and large variants of models

## 🔍 Experimental Setup

- Models tuned for:
  - Epochs: [3, 5, 10]
  - Batch Sizes: [16, 32]
  - Fixed Learning Rate: 2e-4
- Metrics: **F1 (micro & macro)**

### Evaluation Summary

| Model        | F1 Micro | F1 Macro |
|--------------|----------|----------|
| SVM          | 0.22     | 0.21     |
| BERT Base    | 0.43     | 0.34     |
| BERT Large   | 0.52     | 0.41     |
| XLNet Base   | 0.52     | 0.41     |
| XLNet Large  | 0.56     | 0.44     |

## 📈 Key Observations

- **XLNet outperformed all other models**, achieving the highest F1 scores.
- **Transformer models > SVM**, highlighting the importance of contextual representations.
- **Class imbalance** affected macro F1 scores; most models performed better on frequent labels.
- Label **"Universalism: concern"** had the highest individual F1 due to sample prevalence.

## 🔧 Challenges & Improvements

- Computational limitations restricted the hyperparameter search space.
- Performance could improve further by:
  - Balancing the label distribution
  - Expanding the dataset
  - Experimenting with longer sequence lengths and advanced transformer architectures

## 📚 References

- [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [XLNet (Yang et al., 2020)](https://arxiv.org/abs/1906.08237)
- [Schwartz’s Theory of Basic Human Values (2012)](https://doi.org/10.9707/2307-0919.1116)

## 🔗 External Resources

- [Simple Transformers](https://simpletransformers.ai/)
- [Ray Tune for Hyperparameter Optimization](https://docs.ray.io/en/latest/tune/)
- [W&B Sweeps Guide](https://docs.wandb.ai/guides/sweeps)

---

*Project by Ana Slovic, Tony Chahoud, Davide Tarabelloni, and Shadi Farzankia (University of Bologna, Master's in AI)*
