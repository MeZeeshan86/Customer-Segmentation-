import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# Load Data & Model
df = pd.read_csv("Mall_Customers.csv")
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

X = df.iloc[:, [3, 4]].values
pred = kmeans.predict(X)
df["Cluster"] = pred

# App Title & Intro
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown(
    """
    This app uses **K-Means clustering** to group mall customers into 5 segments  
    based on **Annual Income** and **Spending Score**.  
    You can explore the data, see the clusters, and try new inputs below.
    """
)

# Dataset Preview
st.subheader("📂 Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Cluster Distribution
st.subheader("📊 Cluster Distribution")
cluster_counts = df["Cluster"].value_counts().sort_index()
cols = st.columns(len(cluster_counts))
for i, count in enumerate(cluster_counts):
    cols[i].metric(label=f"Cluster {i}", value=int(count))

# Cluster Visualization
st.subheader("🎨 Customer Clusters")
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#FFB347', '#77DD77', '#779ECB', '#CBAACB', '#FF6961']

for i, color in enumerate(colors):
    ax.scatter(
        X[pred == i, 0], X[pred == i, 1],
        c=color, label=f"Cluster {i}",
        alpha=0.7, edgecolors="k", s=50
    )

ax.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=250, c='gold', marker='*', edgecolors="k", label='Centroids'
)

ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Mall Customer Segments", fontsize=12, fontweight="bold")
ax.legend(frameon=True, fontsize=8, loc="best")
plt.tight_layout()
st.pyplot(fig)

# New Input Prediction
st.subheader("➕ Predict a New Customer Segment")
col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=65)
with col2:
    score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, value=40)

if st.button("Predict Cluster 🚀"):
    new_customer = np.array([[income, score]])
    cluster_label = kmeans.predict(new_customer)[0]
    st.success(f"✅ This customer belongs to **Cluster {cluster_label}**")

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, color in enumerate(colors):
        ax.scatter(
            X[pred == i, 0], X[pred == i, 1],
        c=color, label=f"Cluster {i}",
        alpha=0.7, edgecolors="k", s=50
    )

    ax.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        s=250, c='gold', marker='*', edgecolors="k", label='Centroids'
    )

    ax.scatter(
        new_customer[0][0], new_customer[0][1],
        c='black', marker='X', s=200, label='New Customer'
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("New Customer Prediction", fontsize=12, fontweight="bold")
    ax.legend(frameon=True, fontsize=8, loc="best")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(
        f"""
        **Cluster {cluster_label}** is characterized by:
        - **Annual Income**: {income} k$
        - **Spending Score**: {score}
        """
    )