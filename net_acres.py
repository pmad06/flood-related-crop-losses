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

# Aggregate Net_Determined_Acres up to the county level 
# (col is per-record/per-claim, so we sum total damaged acres per county)
acres_by_county = col.groupby('County_Name_clean')['Net_Determined_Acres'].sum().reset_index()
acres_by_county = acres_by_county.rename(columns={'Net_Determined_Acres': 'Total_Determined_Acres'})

# Merge onto the main county dataframe 
df = df.merge(acres_by_county, on='County_Name_clean', how='left')

# Counties with no matching flood records won't have a value 
df['Total_Determined_Acres'] = df['Total_Determined_Acres'].fillna(0)

# checks to verify merge worked
print(df.isna().sum())
print(df[['County_Name', 'Total_Indemnity', 'Total_Determined_Acres']].describe())

# Prepare 2D features: indemnity and total damaged acres
X = df[['Total_Indemnity', 'Total_Determined_Acres']].values
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
chosen_k = 3  # adjust after reviewing results above
final_kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
df['cluster'] = final_kmeans.fit_predict(X_scaled)
df['cluster_display'] = df['cluster'] + 1

print(df.groupby('cluster')[['Total_Indemnity', 'Total_Determined_Acres']].describe())

for c in sorted(df['cluster'].unique()):
    sub = df[df['cluster'] == c].sort_values('Total_Indemnity', ascending=False)
    print(f"\n--- Cluster {c} (n={len(sub)}) ---")
    print(sub[['County_Name', 'Total_Indemnity', 'Total_Determined_Acres']].to_string(index=False))

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
plt.savefig('kmeans_2d_acres_eval.png')
plt.show()

# Scatter plot of the 2D clusters 
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df['Total_Indemnity'], df['Total_Determined_Acres'], c=df['cluster_display'], cmap='viridis', s=60)
ax.set_xlabel('Total Indemnity ($)')
ax.set_ylabel('Total Determined Acres (damaged)')
ax.set_title('2D K-Means Clusters: Indemnity vs. Damaged Acres')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig('kmeans_2d_acres_scatter.png')
plt.show()

# Export
df.to_csv('practice_by_county_2d_acres_clusters.csv', index=False)