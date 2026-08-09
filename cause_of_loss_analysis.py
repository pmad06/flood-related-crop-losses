import matplotlib.pyplot as plt
import pandas as pd

# load data that already has the k-means clusters
df = pd.read_csv('practice_by_county_final_with_clusters.csv')

# load the cause of loss dataset for florida
col = pd.read_csv('florida_flood(in).csv')

# clean up county names and commodity names for merging between the two datasets 
# and grouping by commodity 

col['County_Name_clean'] = col['County_Name'].str.strip()
col['Commodity_Name_clean'] = col['Commodity_Name'].str.strip().str.title()
df['County_Name_clean'] = df['County_Name'].str.strip().str.replace('Miami - Dade', 'Dade')

cluster_means = df.groupby('cluster')['Total_Indemnity'].mean().sort_values()
# low = smallest average, moderate = middle, high = largest
label_map = {
    cluster_means.index[0]: 'Low', 
    cluster_means.index[1]: 'Moderate', 
    cluster_means.index[2]: 'High'
}
df['tier'] = df['cluster'].map(label_map)

# merge two datasets based on county name 
merged = col.merge(df[['County_Name_clean', 'tier']], on='County_Name_clean', how='left')
# filter counties with the high tier
high = merged[merged['tier'] == 'High']
# find total indemnity by commodity for high tier counties
top_commodities = high.groupby('Commodity_Name_clean')['Indemnity'].sum().sort_values(ascending=False).head(8)

# plot the results 

fig, ax = plt.subplots(figsize = (10,6))

top_commodities.plot(kind='barh', ax=ax, color='skyblue')

ax.set_title('Top Commodities Driving Losses in High-Loss Counties', fontsize=14)
ax.set_xlabel('Total Indemnity ($USD)', fontsize=12)
ax.set_ylabel('Commodity', fontsize=12)

plt.tight_layout()
plt.savefig('top_commodities_high_loss_counties.png')
plt.show()