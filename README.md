# 🧠 Patronizing Language Detection

Multi-class classification of condescending and patronizing language targeting vulnerable communities, using the SemEval 2022 PCL dataset from news articles.

## 📌 Overview

This project tackles the task of detecting patronizing and condescending language (PCL) in news articles. The dataset contains paragraphs labelled on a scale of 0–4 based on the degree of patronizing language present, targeting communities such as homeless people, migrants, and people with disabilities.

## 🔍 Approach

- 📊 **Exploratory Analysis** — examined label distributions, class imbalance, and keyword associations across PCL categories
- 🤖 **Baseline Model** — TF-IDF vectorization with a Linear SVM classifier for multi-class prediction
- 🔥 **Deep Learning Model** — fine-tuned DistilBERT with weighted sampling and early stopping to handle severe class imbalance
- 🎯 **Evaluation** — macro-F1 score used as the primary metric to fairly assess performance across all classes

## 📈 Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Linear SVM | — | — |
| DistilBERT (fine-tuned) | 0.77 | 0.39 |

> ⚠️ The gap between accuracy and macro-F1 reflects the challenge of severe class imbalance in the dataset.

## 🛠️ Tech Stack

`Python` `DistilBERT` `Transformers` `scikit-learn` `SVM` `NLP`

## 📂 Dataset

[SemEval 2022 Task 4 — Patronizing and Condescending Language Detection](https://sites.google.com/view/pcl-detection-semeval2022/)
