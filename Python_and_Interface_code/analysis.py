import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def perform_analysis(df):
    plot_dir = "static"
    os.makedirs(plot_dir, exist_ok=True)

    # Drop CustomerID if exists
    df = df.drop(columns=['CustomerID'], errors='ignore')

    # Convert required columns to numeric and drop NaN
    df = df[['MonetaryValue', 'Frequency', 'Recency']].apply(pd.to_numeric, errors='coerce').dropna()

    # Remove outliers beyond 3 standard deviations
    for col in df.columns:
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

    # Scale the data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # K-Means clustering (finding the optimal k)
    inertia, silhouette_scores = [], []
    k_range = range(2, 10)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

        if 1 < k < len(df_scaled):  # Avoid silhouette errors with too few points
            score = silhouette_score(df_scaled, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(None)

    # Debugging: Print silhouette scores
    print("Silhouette Scores:", silhouette_scores)

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))  # Explicitly create a new figure
    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    elbow_path = os.path.join(plot_dir, "elbow_plot.png")
    plt.savefig(elbow_path)
    plt.show()
    plt.close()

    # Plot Silhouette Score
    if any(silhouette_scores):  # Check if scores exist
        plt.figure(figsize=(8, 5))  # Explicitly create a new figure
        plt.plot(k_range[1:], silhouette_scores[1:], marker='o')
        plt.title("Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        silhouette_path = os.path.join(plot_dir, "silhouette_plot.png")
        plt.savefig(silhouette_path)
        plt.show()  # Ensure the graph is displayed
        plt.close()
    else:
        silhouette_path = None  # No valid silhouette scores

    # Find optimal k based on silhouette score
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores[1:])) + 1] if any(silhouette_scores) else 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # 2D Scatter Plot
    plt.figure(figsize=(8, 5))  # Explicitly create a new figure
    scatter = plt.scatter(df['MonetaryValue'], df['Frequency'], c=df['Cluster'], cmap='viridis')
    plt.title(f"K-Means Clustering (MonetaryValue vs Frequency) - {optimal_k} Clusters")
    plt.xlabel("MonetaryValue")
    plt.ylabel("Frequency")
    plt.colorbar(scatter, label="Cluster")
    scatter_2d_path = os.path.join(plot_dir, "scatter_2d_plot.png")
    plt.savefig(scatter_2d_path)
    plt.show()
    plt.close()

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 7))  # Explicitly create a new figure
    ax = fig.add_subplot(111, projection='3d')
    scatter_3d = ax.scatter(df['MonetaryValue'], df['Frequency'], df['Recency'], c=df['Cluster'], cmap='viridis')
    ax.set_title(f"K-Means Clustering (3D) - {optimal_k} Clusters")
    ax.set_xlabel("MonetaryValue")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Recency")
    fig.colorbar(scatter_3d, label="Cluster")
    scatter_3d_path = os.path.join(plot_dir, "scatter_3d_plot.png")
    plt.savefig(scatter_3d_path)
    plt.show()
    plt.close()

    # Cluster Analysis
    def analyze_clusters(dataframe, cols):
        summary = dataframe.groupby('Cluster')[cols].agg(['mean', 'median', 'std', 'count'])
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        return summary

    cluster_summary_2d = analyze_clusters(df, ['MonetaryValue', 'Frequency'])
    cluster_summary_3d = analyze_clusters(df, ['MonetaryValue', 'Frequency', 'Recency'])

    # Save CSV summaries
    cluster_summary_2d_path = os.path.join(plot_dir, "cluster_summary_2d.csv")
    cluster_summary_2d.to_csv(cluster_summary_2d_path)

    cluster_summary_3d_path = os.path.join(plot_dir, "cluster_summary_3d.csv")
    cluster_summary_3d.to_csv(cluster_summary_3d_path)

    # Display all plots together in a single figure
    plt.figure(figsize=(15, 10))  # Create a large figure for all plots

    # Elbow Method Plot
    plt.subplot(2, 2, 1)  # 2 rows, 2 columns, position 1
    plt.plot(k_range, inertia, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    # Silhouette Score Plot
    if any(silhouette_scores):
        plt.subplot(2, 2, 2)  # 2 rows, 2 columns, position 2
        plt.plot(k_range[1:], silhouette_scores[1:], marker='o')
        plt.title("Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")

    # 2D Scatter Plot
    plt.subplot(2, 2, 3)  # 2 rows, 2 columns, position 3
    scatter = plt.scatter(df['MonetaryValue'], df['Frequency'], c=df['Cluster'], cmap='viridis')
    plt.title(f"K-Means Clustering (MonetaryValue vs Frequency) - {optimal_k} Clusters")
    plt.xlabel("MonetaryValue")
    plt.ylabel("Frequency")
    plt.colorbar(scatter, label="Cluster")

    # 3D Scatter Plot
    ax = plt.subplot(2, 2, 4, projection='3d')  # 2 rows, 2 columns, position 4
    scatter_3d = ax.scatter(df['MonetaryValue'], df['Frequency'], df['Recency'], c=df['Cluster'], cmap='viridis')
    ax.set_title(f"K-Means Clustering (3D) - {optimal_k} Clusters")
    ax.set_xlabel("MonetaryValue")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Recency")
    plt.colorbar(scatter_3d, label="Cluster")

    # Adjust layout and display
    plt.tight_layout()
    combined_plot_path = os.path.join(plot_dir, "combined_plot.png")
    plt.savefig(combined_plot_path)
    plt.show()

    return {
        "elbow_plot": elbow_path,
        "silhouette_plot": silhouette_path,
        "scatter_2d_plot": scatter_2d_path,
        "scatter_3d_plot": scatter_3d_path,
        "cluster_summary_2d": cluster_summary_2d_path,
        "cluster_summary_3d": cluster_summary_3d_path,
        "combined_plot": combined_plot_path,
    }


# Example usage:
# df = pd.read_csv("your_data.csv")  # Load your dataset
# results = perform_analysis(df)