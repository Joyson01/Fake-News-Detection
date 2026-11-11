# News Classification Project

A Python learning project for exploring news classification with TF-IDF features and scikit-learn models: Logistic Regression, Support Vector Classifier, and Random Forest.

## Current status

The script and the previous documentation have diverged. This repository is an experiment that needs corrections before its evaluation results can be relied on:

- `main.py` creates a `label` column, but selects `data['News']` as the target. Inputs without a `News` column will fail at that point.
- The binary precision, recall, and F1 calls use their default positive-label setting; this needs to match the actual target encoding.
- TF-IDF is fitted before the train/test split, allowing information from the test set into the feature vocabulary.
- The Random Forest is saved unconditionally as `best_model.pkl`; the script does not select the best model by comparing scores, and it does not save the fitted vectorizer with the model.

These are documented limitations, not fixes. Predictions from this experiment should not be treated as fact verification.

## Environment

Use Python 3 with Tkinter and a graphical desktop for file selection and plots.

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn nltk contractions
```

`pickle` is part of Python's standard library; it does not need a separate package.

The text-processing code uses NLTK's English stopwords, word tokenizer, and WordNet resources. Install the resources requested by NLTK if a lookup error occurs; tokenizer resource names can depend on the NLTK version.

## Entry point and data

```bash
git clone https://github.com/Joyson01/Fake-News-Detection.git
cd Fake-News-Detection
python main.py
```

The script first asks for the true-news CSV, then the fake-news CSV. It reads `title` and `text` fields; the target-column inconsistency above must be resolved for a complete run. Close each plot window to let the script continue.

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | Preprocessing, feature extraction, training, evaluation, and model serialization |
| `Data/` | CSV files for the experiment |
| `best_model.pkl` | Output written by a completed run; not a complete inference pipeline |

## Author and acknowledgments

**Joyson Vadlya**

Inspired by online tutorials and machine-learning resources.
