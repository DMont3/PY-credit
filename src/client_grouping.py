import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib
import logging
import argparse
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_preprocessed_data(file_path: str):
    df = pd.read_csv(file_path)
    logging.info(f"Loaded data shape: {df.shape}")
    return df


def determine_optimal_clusters(data: np.ndarray, max_clusters: int = 10) -> int:
    inertias = []
    silhouette_scores = []

    for k in range(2, min(max_clusters + 1, len(data))):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, min(max_clusters + 1, len(data))), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, min(max_clusters + 1, len(data))), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300)
    plt.close()

    optimal_clusters = np.argmin(np.diff(inertias)) + 2
    return optimal_clusters


def cluster_clients(data: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)


def train_group_classifier(data: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    logging.info(f"Group classifier accuracy: {accuracy:.2f}")
    return clf


def main(input_path: str, output_path: str):
    try:
        df = load_preprocessed_data(input_path)
        logging.info(f"Data shape after preprocessing: {df.shape}")

        # Use all columns for clustering
        X = df.values

        optimal_clusters = determine_optimal_clusters(X)
        logging.info(f"Optimal number of clusters: {optimal_clusters}")

        cluster_labels = cluster_clients(X, optimal_clusters)
        group_classifier = train_group_classifier(X, cluster_labels)

        # Add cluster labels to the original dataframe
        df['cluster'] = cluster_labels

        # Save the classifier and clustered data
        joblib.dump(group_classifier, output_path)
        df.to_csv(os.path.splitext(output_path)[0] + '_clustered_data.csv', index=False)

        logging.info(f"Client grouping completed successfully. Classifier saved to {output_path}")
        logging.info(f"Clustered data saved to {os.path.splitext(output_path)[0] + '_clustered_data.csv'}")
    except Exception as e:
        logging.error(f"An error occurred during client grouping: {str(e)}")
        logging.error("Error details: ", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client Grouping Script")
    parser.add_argument("--input", required=True, help="Path to the preprocessed input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the group classifier")
    args = parser.parse_args()

    main(args.input, args.output)