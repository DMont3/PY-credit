import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc, \
    silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import argparse
import os

from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_and_preprocess_data(file_path: str):
    # Use the same function as in preprocessamento.py
    from preprocessamento import load_and_preprocess_data as load_data
    df, numeric_columns, categorical_columns, boolean_columns = load_data(file_path)

    # Select features for clustering (you may need to adjust this based on your specific needs)
    features_for_clustering = numeric_columns + boolean_columns
    X = df[features_for_clustering]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df


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
        X_scaled, df = load_and_preprocess_data(input_path)
        logging.info(f"Data shape after preprocessing: {X_scaled.shape}")

        optimal_clusters = determine_optimal_clusters(X_scaled)
        logging.info(f"Optimal number of clusters: {optimal_clusters}")

        cluster_labels = cluster_clients(X_scaled, optimal_clusters)
        group_classifier = train_group_classifier(X_scaled, cluster_labels)

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
    parser.add_argument("--input", required=True, help="Path to the input CSV file")
    parser.add_argument("--output", required=True, help="Path to save the group classifier")
    args = parser.parse_args()

    main(args.input, args.output)


def train_classifier(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Evaluation on test set
    y_pred = clf.predict(X_test)
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))

    return clf


def train_regressor(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    logging.info(f"Cross-validation RMSE scores: {rmse_scores}")
    logging.info(f"Mean CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")

    # Evaluation on test set
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"\nTest set MSE: {mse:.4f}")
    logging.info(f"Test set R2 score: {r2:.4f}")

    return reg


def plot_feature_importance(model, feature_names: list, title: str):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16)
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Importance', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def plot_confusion_matrix(clf: RandomForestClassifier, X: pd.DataFrame, y: pd.Series, title: str):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def plot_roc_curve(clf: RandomForestClassifier, X: pd.DataFrame, y: pd.Series, title: str):
    y_pred_proba = clf.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def plot_regression_results(reg: RandomForestRegressor, X: pd.DataFrame, y: pd.Series, title: str):
    y_pred = reg.predict(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


def main(input_path: str, output_dir: str):
    try:
        # Load preprocessed data
        X = pd.read_csv(input_path)
        preprocessor = joblib.load(os.path.splitext(input_path)[0] + '_preprocessor.joblib')

        # Load original data to get target variables
        original_data_path = os.path.splitext(input_path)[0].replace('_preprocessed', '') + '.csv'
        original_data = pd.read_csv(original_data_path)
        y_classificacao = original_data['status']
        y_regressao = original_data['valorAprovado']

        # Train and evaluate classifier
        logging.info("Training classifier...")
        clf = train_classifier(X, y_classificacao)
        plot_feature_importance(clf, X.columns, "Classification Feature Importance")

        # Train and evaluate regressor
        logging.info("\nTraining regressor...")
        reg = train_regressor(X, y_regressao)
        plot_feature_importance(reg, X.columns, "Regression Feature Importance")

        # Generate and save additional visualizations
        plot_confusion_matrix(clf, X, y_classificacao, "Confusion Matrix - Classification")
        plot_roc_curve(clf, X, y_classificacao, "ROC Curve - Classification")
        plot_regression_results(reg, X, y_regressao, "Regression Results")

        # Perform cross-validation and display results
        cv_results_clf = cross_validate(clf, X, y_classificacao, cv=5, scoring=['accuracy', 'f1_weighted', 'roc_auc'])
        cv_results_reg = cross_validate(reg, X, y_regressao, cv=5, scoring=['r2', 'neg_mean_squared_error'])

        logging.info("\nCross-validation results - Classification:")
        logging.info(
            f"Accuracy: {cv_results_clf['test_accuracy'].mean():.4f} (+/- {cv_results_clf['test_accuracy'].std() * 2:.4f})")
        logging.info(
            f"F1-weighted: {cv_results_clf['test_f1_weighted'].mean():.4f} (+/- {cv_results_clf['test_f1_weighted'].std() * 2:.4f})")
        logging.info(
            f"ROC AUC: {cv_results_clf['test_roc_auc'].mean():.4f} (+/- {cv_results_clf['test_roc_auc'].std() * 2:.4f})")

        logging.info("\nCross-validation results - Regression:")
        logging.info(
            f"R2 Score: {cv_results_reg['test_r2'].mean():.4f} (+/- {cv_results_reg['test_r2'].std() * 2:.4f})")
        logging.info(
            f"MSE: {-cv_results_reg['test_neg_mean_squared_error'].mean():.4f} (+/- {cv_results_reg['test_neg_mean_squared_error'].std() * 2:.4f})")

        # Save models
        joblib.dump(clf, os.path.join(output_dir, 'classificador_modelo.joblib'))
        joblib.dump(reg, os.path.join(output_dir, 'regressor_modelo.joblib'))
        joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))

        logging.info("\nModels and preprocessor saved.")

    except Exception as e:
        logging.error(f"An error occurred during model training: {str(e)}")
        logging.error("Error details: ", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument("--input", required=True, help="Path to the preprocessed input CSV file")
    parser.add_argument("--output", required=True, help="Directory to save the trained models")
    args = parser.parse_args()

    main(args.input, args.output)