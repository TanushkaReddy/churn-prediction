import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/cleaned_data.csv')

# ── Build RFM Table ──────────────────────────────────────────────────────────
rfm = df[['CustomerID', 'DaySinceLastOrder', 'OrderCount',
          'CashbackAmount', 'Churn']].copy()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Churn']

# Lower recency = more recent = better, so we invert it
rfm['Recency'] = rfm['Recency'].max() - rfm['Recency']

print("RFM Table Sample:")
print(rfm.head())
print(f"\nRFM Shape: {rfm.shape}")

# ── RFM Scoring (1-5 scale) ──────────────────────────────────────────────────
rfm['R_Score'] = pd.qcut(rfm['Recency'],  q=5, labels=[1,2,3,4,5])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5])

rfm['RFM_Score'] = (rfm['R_Score'].astype(int) +
                    rfm['F_Score'].astype(int) +
                    rfm['M_Score'].astype(int))

print("\nRFM Score Distribution:")
print(rfm['RFM_Score'].describe())

# ── Find Optimal K using Elbow Method ───────────────────────────────────────
features = ['Recency', 'Frequency', 'Monetary']
scaler   = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

inertias    = []
sil_scores  = []
K_range     = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km.labels_))

# Plot elbow
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method — Finding Optimal K')
axes[0].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
axes[0].legend()

axes[1].plot(K_range, sil_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by K')
axes[1].axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
axes[1].legend()

plt.tight_layout()
plt.savefig('data/elbow_plot.png', dpi=150)
plt.show()
print("Elbow plot saved!")

# ── Apply K-Means with K=4 ───────────────────────────────────────────────────
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# ── Analyze Clusters ─────────────────────────────────────────────────────────
cluster_summary = rfm.groupby('Cluster').agg(
    Recency   =('Recency',   'mean'),
    Frequency =('Frequency', 'mean'),
    Monetary  =('Monetary',  'mean'),
    Churn_Rate=('Churn',     'mean'),
    Count     =('CustomerID','count')
).round(2)

print("\nCluster Summary:")
print(cluster_summary)

# ── Name the segments ────────────────────────────────────────────────────────
def label_segment(row):
    if row['Churn_Rate'] > 0.3:
        return 'High Risk'
    elif row['Monetary'] > rfm['Monetary'].median() and row['Frequency'] > rfm['Frequency'].median():
        return 'Champions'
    elif row['Recency'] > rfm['Recency'].median():
        return 'Loyal Customers'
    else:
        return 'At Risk'

cluster_summary['Segment'] = cluster_summary.apply(label_segment, axis=1)
print("\nSegment Labels:")
print(cluster_summary[['Segment', 'Churn_Rate', 'Count']])

# Map segment names back to rfm df
segment_map = cluster_summary['Segment'].to_dict()
rfm['Segment'] = rfm['Cluster'].map(segment_map)

# ── Visualize Segments ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Recency by segment
sns.boxplot(ax=axes[0], x='Segment', y='Recency', data=rfm,
            hue='Segment', palette='Set2', legend=False)
axes[0].set_title('Recency by Segment')
axes[0].tick_params(axis='x', rotation=15)

# Frequency by segment
sns.boxplot(ax=axes[1], x='Segment', y='Frequency', data=rfm,
            hue='Segment', palette='Set2', legend=False)
axes[1].set_title('Frequency by Segment')
axes[1].tick_params(axis='x', rotation=15)

# Monetary by segment
sns.boxplot(ax=axes[2], x='Segment', y='Monetary', data=rfm,
            hue='Segment', palette='Set2', legend=False)
axes[2].set_title('Monetary by Segment')
axes[2].tick_params(axis='x', rotation=15)

plt.suptitle('RFM Segments — Behavior Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('data/rfm_segments.png', dpi=150)
plt.show()
print("RFM segments plot saved!")

# ── Churn Rate by Segment ────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
churn_by_seg = rfm.groupby('Segment')['Churn'].mean().sort_values(ascending=False)
sns.barplot(x=churn_by_seg.index, y=churn_by_seg.values,
            hue=churn_by_seg.index, palette='Reds_r', legend=False)
plt.title('Churn Rate by Customer Segment')
plt.xlabel('Segment')
plt.ylabel('Churn Rate')
plt.tight_layout()
plt.savefig('data/churn_by_segment.png', dpi=150)
plt.show()
print("Churn by segment plot saved!")

# ── Save RFM results ─────────────────────────────────────────────────────────
rfm.to_csv('data/rfm_segments.csv', index=False)
print("\nRFM data saved as rfm_segments.csv")
print("\nPhase 5 Complete!")