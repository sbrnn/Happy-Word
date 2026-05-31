# HappinessMap 🌍

Unsupervised machine learning project that clusters countries 
based on their happiness indicators using PCA and KMeans.

## Dataset
World Happiness Report dataset from Kaggle
- 156 countries
- Features: GDP, Social Support, Health, Freedom, 
  Generosity, Corruption

## Pipeline
1. Load & explore the data
2. Select relevant features
3. Scale the data (StandardScaler)
4. Reduce dimensions (PCA → 2D)
5. Find optimal K (Elbow Method)
6. Cluster countries (KMeans)
7. Visualize clusters with country labels

## Technologies
- Python
- Pandas
- Scikit-learn
- Matplotlib

## How to Run
1. Clone the repo
2. Download dataset from Kaggle
3. Run `happiness_map.ipynb`

## Results
Countries are grouped into 3 clusters representing
high, medium, and low happiness profiles.

## Data Source
World Happiness Report
https://www.kaggle.com/datasets/unsolved/world-happiness-report