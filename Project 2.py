import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load Dataset:
data = pd.read_csv("C:/Users/Abdullah Umer/Desktop/Data Zenix Solutions Internship/Project 2/Customer_ Segmentation_ Data.csv")

print("☑️ First 5 rows:\n", data.head())
print("\n☑️ Dataset Info:\n")
print(data.info())

# 2. Data Cleaning and Exploration:
# Remove duplicates if any
data.drop_duplicates(inplace=True)

# 3. Descriptive Statistics:
print("\n☑️ Descriptive Statistics:\n", data.describe())

# Example metrics
avg_purchase = data['last_purchase_amount'].mean()
avg_freq = data['purchase_frequency'].mean()
print(f"☑️ Average Purchase Amount: {avg_purchase:.2f}")
print(f"☑️ Average Purchase Frequency: {avg_freq:.2f}")

# 4. Customer Segmentation (K-Means Clustering):
# Select features for clustering
features = ['age', 'income', 'spending_score', 'purchase_frequency', 'last_purchase_amount']
X = data[features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using Elbow Method
inertia = []
K = range(2, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choose k=5 for example (can adjust based on elbow plot)
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Evaluate clustering with silhouette score
score = silhouette_score(X_scaled, data['Cluster'])
print(f"☑️ Silhouette Score: {score:.2f}")

# 5. Visualization of Customer Segments:
sns.set(style="whitegrid")

plt.figure(figsize=(8,6))
sns.scatterplot(x='income', y='spending_score', hue="Cluster", data=data, palette="Set2")
plt.title("Customer Segments by Income vs Spending Score")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='Cluster', hue='Cluster', data=data, palette="Set2", legend=False)
plt.title("Number of Customers in Each Cluster")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(data['age'], bins=20, kde=True, color="yellow")
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='gender', y='income', hue='gender', data=data, palette="Set3", legend=False)
plt.title("Income Distribution by Gender")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='preferred_category', hue='preferred_category', data=data,
              order=data['preferred_category'].value_counts().index,
              palette="muted", legend=False)
plt.title("Most Preferred Product Categories")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='spending_score', y='purchase_frequency', hue='Cluster', data=data, palette="Set1")
plt.title("Spending Score vs Purchase Frequency by Cluster")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(x='Cluster', y='last_purchase_amount', hue='Cluster', data=data,
            errorbar=None, palette="coolwarm", legend=False)
plt.title("Average Last Purchase Amount per Cluster")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='membership_years', y='spending_score', hue='Cluster', data=data, palette="Dark2")
plt.title("Spending Score vs Membership Years by Cluster")
plt.show()

# 6. Insights & Recommendations:
cluster_summary = data.groupby('Cluster')[features].mean()
print("\n☑️ Cluster Summary (Average Values):\n", cluster_summary)

print("\n☑️ Recommendations:")
for cluster_id, row in cluster_summary.iterrows():
    print(f"- Cluster {cluster_id}: avg income={row['income']:.2f}, spending_score={row['spending_score']:.2f}, freq={row['purchase_frequency']:.2f}")
    if row['spending_score'] > 70:
        print("  -> High-value customers, target with premium offers.")
    elif row['spending_score'] < 30:
        print("  -> Low-engagement customers, offer discounts to retain.")
    else:
        print("  -> Medium-value customers, nurture with loyalty programs.")








