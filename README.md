# Customer Reviews Analysis: NLP, Time Series, and Neural Networks

Summative lab for an analytics firm client. The notebook analyzes Amazon customer reviews of e-reader devices (Nook, Kindle) in three parts.

## Notebook

`customer_reviews_analysis.ipynb`

| Part | Topic | What it does |
|------|-------|--------------|
| 1 | NLP | Cleans review text, then finds the most frequent words and bigrams and visualizes them. |
| 2 | Time Series | Resamples reviews to monthly counts and ratings, tests stationarity (ADF/KPSS), and fits and compares ARIMA models against a naive forecast. |
| 3 | Neural Networks | Builds TF-IDF features and trains a Keras neural network to predict star ratings, compared with baselines and several network variants. |

Each part ends with a written reflection summarizing findings and limitations.

## Dataset

The notebook expects `amazon_reviews_lab.csv` in the same folder, with at least `review_text`, `rating`, and `timestamp` columns. The file is not included in this repository (it is listed in `.gitignore`).

## Setup

```bash
git clone https://github.com/ainohamjeptoo-cloud/customer-reviews-analysis.git
cd customer-reviews-analysis
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook customer_reviews_analysis.ipynb
```

The first NLP cell downloads the required NLTK resources automatically.

## Tools

Python, pandas, NumPy, Matplotlib, NLTK, scikit-learn, statsmodels, TensorFlow/Keras.
