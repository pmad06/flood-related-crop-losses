import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('practice_by_county_2d_acres_clusters.csv')

fig, ax = plt.subplots(figsize=(6, 6))
cluster_counts = df['cluster'].value_counts().sort_index()
ax.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index],
       autopct='%1.1f%%', colors=['#f4a582', '#b2182b', '#92c5de'])
ax.set_title('Proportion of Florida Counties by Cluster')
plt.tight_layout()
plt.savefig('pie_chart_cluster_counts.png', dpi=150)