import pandas as pd
import matplotlib.pyplot as plt

# Load your 2D clustered data
df = pd.read_csv('practice_by_county_2d_acres_clusters.csv')

df['cluster_display'] = df['cluster'] + 1

# Box plot: distribution of Total_Indemnity within each cluster
fig, ax = plt.subplots(figsize=(8, 6))
df.boxplot(column='Total_Indemnity', by='cluster_display', ax=ax)
ax.set_xlabel('Cluster')
ax.set_ylabel('Total Indemnity ($)')
ax.set_title('Distribution of Total Indemnity by Cluster')
plt.suptitle('')  # removes the default pandas subtitle, which looks messy
plt.tight_layout()
plt.savefig('boxplot_indemnity_by_cluster.png', dpi=150)
plt.show()