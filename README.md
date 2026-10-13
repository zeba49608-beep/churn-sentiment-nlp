# churn-sentiment-nlp
# NLP-Based Customer Sentiment Analysis for Churn Prediction

An end-to-end pipeline that extracts sentiment from customer feedback with NLP, compares classical and deep learning classifiers, and merges the sentiment features with structured churn data to test whether text improves churn prediction.

## Objective
Analyze customer feedback with NLP, generate sentiment insights, and merge them with structured churn data (`Churn_Modeling.csv`) to improve churn prediction performance.

## Datasets
| File | Description |
|------|-------------|
| `Churn_Modeling.csv` | Structured customer attributes and the churn label `Exited` |
| `Customer_Feedback.csv` | Free-text feedback and star ratings per `CustomerId` |

Place both files in `data/` (they are git-ignored by default).

## Pipeline
1. **Data loading and exploration**: structure, dtypes, missing values, rating distribution, text length.
2. **Text preprocessing**: lowercasing, punctuation removal, tokenization, stopword removal (negations kept), lemmatization (NLTK).
3. **Sentiment labeling**: Positive / Neutral / Negative from star ratings (4-5 / 3 / 1-2); VADER and TextBlob polarity computed as continuous features.
4. **Vectorization**: TF-IDF (unigrams + bigrams) and Bag-of-Words.
5. **Modeling and evaluation**: Logistic Regression, Linear SVM and an LSTM, compared on accuracy, precision, recall and F1.
6. **Integration**: per-customer sentiment features (mean polarity, share of negative/positive feedback, feedback count, average rating) merged with churn data on `CustomerId`.
7. **Visualization and insights**: sentiment distribution, polarity histogram, word clouds per sentiment, correlation of sentiment with `Exited`, churn rate by polarity quintile, and a churn model with vs. without sentiment features.

## Repository structure
```
churn-sentiment-nlp/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # put the input CSVs here
├── notebooks/
│   └── churn_sentiment_analysis.ipynb
├── src/
│   └── churn_sentiment_pipeline.py
└── outputs/               # generated plots and tables
```

## How to run
```bash
git clone https://github.com/zeba49608-beep/churn-sentiment-nlp.git
cd churn-sentiment-nlp
pip install -r requirements.txt
# add the CSVs to data/
python src/churn_sentiment_pipeline.py      # or open the notebook
```
**Google Colab:** upload the notebook, upload the CSVs, and edit `CHURN_PATH` / `FEEDBACK_PATH` in the config cell to match where the files are.

## Results
> Fill these in from your own run.

**Sentiment classifier comparison**

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| LogReg (TF-IDF) | | | | |
| LinearSVM (TF-IDF) | | | | |
| LogReg (BoW) | | | | |
| LSTM | | | | |

**Churn model with vs. without sentiment**

| Feature set | ROC-AUC | F1 |
|-------------|---------|----|
| Structured only | | |
| Structured + sentiment | | |

## Key insights
- *(Relationship between sentiment polarity and churn, from the correlation plot and quintile chart)*
- *(Which model performed best and why)*
- *(Common words in negative feedback, from the word clouds)*

Plots are saved to `outputs/`.

## Limitations
- Sentiment labels come from star ratings, and sentiment predictions in the merge step are made on data the model was trained on, so the sentiment features may be optimistic.
- Not every customer has feedback; missing sentiment features are filled with 0 in the churn model.
- Small text datasets favor simple TF-IDF models over LSTMs.

## Tech stack
Python, pandas, scikit-learn, NLTK, VADER, TextBlob, TensorFlow/Keras, WordCloud, matplotlib, seaborn.
