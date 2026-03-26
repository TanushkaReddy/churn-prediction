import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import pickle
warnings.filterwarnings('ignore')
df = pd.read_csv('data/cleaned_data.csv')

# ── Encode categorical columns ──────────────────────────────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ── Features & target ───────────────────────────────────────────────────────
X = df.drop(columns=['Churn', 'CustomerID'])
y = df['Churn']

# ── Train / test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Churn rate in train: {y_train.mean():.2%}")

# ── Handle class imbalance with sample weights ───────────────────────────────
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# ── XGBoost model ────────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)
print("\nModel trained successfully!")

# ── Evaluation ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ── Cross validation ─────────────────────────────────────────────────────────
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"\n5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── Confusion Matrix ─────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('data/confusion_matrix.png', dpi=150)
plt.show()
print("Confusion matrix saved!")

# ── ROC Curve ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.4f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('data/roc_curve.png', dpi=150)
plt.show()
print("ROC curve saved!")

# ── SHAP Feature Importance ──────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.title('SHAP Feature Importance')
plt.tight_layout()
plt.savefig('data/shap_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP plot saved!")

# ── Save model ───────────────────────────────────────────────────────────────
with open('data/churn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

feature_names = X.columns.tolist()
with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("\nModel saved as churn_model.pkl")
print("Feature names saved as feature_names.pkl")
print("\nPhase 4 Complete!")