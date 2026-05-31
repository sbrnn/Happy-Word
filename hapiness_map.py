import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("world_happiness.csv")
print(df.head())

features = ['GDP per capita', 'Social support',
            'Healthy life expectancy',
            'Freedom to make life choices',
            'Generosity',
            'Perceptions of corruption']

X = df[features]
countries = df['Country or region']

# Drop missing values
X.dropna(inplace=True)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance explained:", pca.explained_variance_ratio_)

# Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# KMeans with optimal K
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')

# Add country labels
for i, country in enumerate(countries):
    plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]), fontsize=6)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('World Happiness Clusters')
plt.show()

# Cluster profiles
df['cluster'] = clusters
print(df.groupby('cluster')[features].mean())