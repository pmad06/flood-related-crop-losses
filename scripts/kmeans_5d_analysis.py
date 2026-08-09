import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('practice_by_county_final(in).csv')
col = pd.read_csv('florida_flood(in).csv')

# Clean county names for merging
df['County_Name_clean'] = df['County_Name'].str.strip().str.replace('Miami - Dade', 'Dade')
col['County_Name_clean'] = col['County_Name'].str.strip()

df['Pct_Irrigated'] = df['Pct_Irrigated'].fillna(0)

# Aggregate multiple variables from the cause-of-loss data up to county level 
county_agg = col.groupby('County_Name_clean').agg(
    Total_Determined_Acres=('Net_Determined_Acres', 'sum'),
    Net_Planted_Acres=('Net_Planted_Acres', 'sum'),
    Loss_Ratio=('Loss_Ratio', 'mean'),          # mean automatically skips NaNs
    Policies_Indemnified=('Policies_Indemnified', 'sum')
).reset_index()

# Merge onto the main county dataframe
df = df.merge(county_agg, on='County_Name_clean', how='left')

# Counties with no matching flood records won't have a value 
agg_cols = ['Total_Determined_Acres', 'Net_Planted_Acres', 'Loss_Ratio', 'Policies_Indemnified']
df[agg_cols] = df[agg_cols].fillna(0)

# checks to verify merge worked
print(df.isna().sum())
print(df[['Total_Indemnity', 'Total_Determined_Acres', 'Pct_Irrigated',
          'Net_Planted_Acres', 'Loss_Ratio', 'Policies_Indemnified']].describe())

# Prepare 5D features
feature_cols = ['Total_Indemnity', 'Total_Determined_Acres', 'Pct_Irrigated',
                 'Net_Planted_Acres', 'Loss_Ratio']
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test k=2 through 8
results = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    results.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_score(X_scaled, labels)
    })

results_df = pd.DataFrame(results)
print(results_df)

best_k = results_df.loc[results_df['silhouette_score'].idxmax(), 'k']
print(f'Best k by silhouette score: {best_k}')

# Fit final model
chosen_k = 2  # adjust after reviewing results above
final_kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
df['cluster'] = final_kmeans.fit_predict(X_scaled)

print(df.groupby('cluster')[feature_cols].describe())

for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c].sort_values('Total_Indemnity', ascending=False)
    print(f"\n--- Cluster {c} (n={len(sub)}) ---")
    print(sub[['County_Name'] + feature_cols].to_string(index=False))

# Plot elbow and silhouette models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(results_df['k'], results_df['inertia'], marker='o')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')
axes[1].plot(results_df['k'], results_df['silhouette_score'], marker='o', color='orange')
axes[1].set_title('Silhouette Method')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig('kmeans_5d_eval.png')

# plot variables colored by the 5d clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['Total_Indemnity'], df['Total_Determined_Acres'], c=df['cluster'], cmap='viridis', s=60)
ax.set_xlabel('Total Indemnity ($)')
ax.set_ylabel('Total Determined Acres (damaged)')
ax.set_title('5D K-Means Clusters (visualized on 2 of 5 variables)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_5d_scatter.png')

# Box plot of indemnity distribution by cluster
fig, ax = plt.subplots(figsize=(8, 6))
df.boxplot(column='Total_Indemnity', by='cluster', ax=ax)
ax.set_xlabel('Cluster')
ax.set_ylabel('Total Indemnity ($)')
ax.set_title('Distribution of Total Indemnity by Cluster (5D clustering)')
plt.suptitle('')
plt.tight_layout()
plt.savefig('kmeans_5d_boxplot.png')

# Export csv files
df.to_csv('practice_by_county_5d_clusters.csv', index=False)