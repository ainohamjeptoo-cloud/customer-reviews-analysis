# Customer Reviews Analysis: NLP, Time Series, and Neural Networks

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-red)

An end-to-end analysis of Amazon customer reviews of e-reader devices (Nook and Kindle), built as a summative lab for an analytics firm client. The project combines three techniques to understand what customers talk about, how review activity changes over time, and whether star ratings can be predicted from review text.

## Table of Contents

- [Business Questions](#business-questions)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Tools and Libraries](#tools-and-libraries)
- [Author](#author)

## Business Questions

1. **NLP:** Which topics, products, and features do customers mention most often?
2. **Time series:** How do review volume and ratings change over time, and can review volume be forecast?
3. **Neural networks:** Can a model predict a review's star rating from its text?

## Project Structure

```
customer-reviews-analysis-lab/
├── customer_reviews_analysis.ipynb   # Full analysis with written reflections
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excludes the dataset, caches, and environments
├── README.md
└── amazon_reviews_lab.csv            # Not tracked; add locally (see Dataset)
```

## Methods

| Part | Topic | Approach |
|------|-------|----------|
| 1 | NLP | Text cleaning (lowercasing, URL and HTML removal, tokenization, stopword removal, lemmatization), word and bigram frequency counts, and bar chart visualizations. |
| 2 | Time Series | Monthly resampling of review counts and average ratings, 3-month rolling average, ADF and KPSS stationarity tests, ACF/PACF plots, and ARIMA models compared against a naive forecast on a chronological train/test split. |
| 3 | Neural Networks | TF-IDF features (3,000 unigrams and bigrams) feeding a Keras multilayer perceptron. Compared with majority-class and logistic regression baselines, then tested with class weights, L2 regularization, dropout, early stopping, and different optimizers. |

Each part ends with a written reflection on findings and limitations.

## Key Results

**Part 1: NLP**
- The most frequent words were "nook", "book", and "kindle", so customers mainly discuss reading devices and books.
- Common bigrams include "barnes noble", "nook color", "battery life", "touch screen", and "customer service". These point to specific devices and features that a business may want to investigate.
- Word frequency cannot show whether customers were satisfied, so it should be read alongside ratings or sentiment.

**Part 2: Time Series**
- Reviews before 2010 are sparse (16 in total), so modeling uses January 2010 to June 2014 (54 months). July 2014 is excluded because the data ends mid-month.
- Monthly volume averages about 18 reviews (range 3 to 46) and peaks around the holiday periods.
- ADF and KPSS gave conflicting results on the raw series, which points to a trend, so the models use log and differencing transformations.

**Part 3: Neural Networks**
- About 58% of test reviews are 5 stars, so accuracy alone is misleading. Models are compared on macro F1 and per-class recall.
- The original network scored 54.5% accuracy (macro F1 0.229), below the 58.0% majority-class baseline.
- The best tuned network reached 57.0% accuracy and 0.321 macro F1, which is close to balanced logistic regression (0.315). It still overfit, with 95.0% training accuracy against 56.2% validation accuracy.
- SGD diverged at a high learning rate and underfit at a lower one, so Adam worked better on this sparse input.

## Dataset

The notebook expects `amazon_reviews_lab.csv` in the same folder as the notebook, with at least these columns:

| Column | Description |
|--------|-------------|
| `review_text` | Text of the customer review |
| `rating` | Star rating from 1 to 5 |
| `timestamp` | Date of the review |

The file is not included in this repository (it is listed in `.gitignore`). Add your copy before running the notebook.

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/ainohamjeptoo-cloud/customer-reviews-analysis-lab.git
cd customer-reviews-analysis-lab

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add amazon_reviews_lab.csv to this folder, then start Jupyter
jupyter notebook customer_reviews_analysis.ipynb
```

Run the cells from top to bottom. The first NLP cell downloads the NLTK resources it needs (punkt, stopwords, wordnet, omw-1.4).

## Limitations

- The dataset is small (797 training reviews for the neural network), which limits what the models can learn.
- Ratings are heavily skewed toward 5 stars, so minority classes are hard to predict.
- Word and bigram counts describe topics, not sentiment.
- The forecast is based on a short series (54 months), so confidence intervals are wide.

## Tools and Libraries

Python, pandas, NumPy, Matplotlib, NLTK, scikit-learn, statsmodels, TensorFlow/Keras, Jupyter.

## Author

Ainoham Jeptoo
