# E-Commerce Customer Churn Prediction

> Predict · Segment · Retain — powered by XGBoost & RFM Analysis

## Results
| Metric | Score |
|---|---|
| ROC-AUC | 0.9969 |
| 5-Fold CV AUC | 0.9984 |
| Accuracy | 98% |
| Recall (Churn) | 99% |

## Tech Stack
Python · XGBoost · SHAP · Scikit-learn · Streamlit · Plotly · MySQL · Pandas

## Project Structure
```
churn-prediction/
├── data/               # Dataset & saved models
├── notebooks/          # EDA notebook
├── src/
│   ├── preprocess.py   # Data cleaning
│   ├── model.py        # XGBoost training
│   ├── rfm.py          # RFM segmentation
│   └── db.py           # MySQL integration
└── app.py              # Streamlit dashboard
```

## How to Run
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Key Findings
- Low tenure customers (< 6 months) are 3x more likely to churn
- Complaints increase churn probability by 25%
- Higher cashback significantly reduces churn risk