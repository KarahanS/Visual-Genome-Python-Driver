import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def prepare_features(data):
    """
    Prepare features with square root transformation
    """
    X = pd.DataFrame(
        {
            "sqrt_num_seg": np.sqrt(data["num_seg"]),
            "sqrt_num_class": np.sqrt(data["num_class"]),
        }
    )
    return X


def evaluate_fold(X_train, y_train, X_test, y_test):
    """
    Evaluate one fold of cross-validation
    """
    # Add constant for intercept
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit model
    model = OLS(y_train, X_train_const)
    results = model.fit()

    # Make predictions
    y_pred = results.predict(X_test_const)

    # Calculate Spearman correlation
    correlation, _ = spearmanr(y_test, y_pred)

    return correlation, results


def cross_validate_regression(X, y, num_repeats, n_splits=3):
    """
    Perform repeated k-fold cross-validation
    """
    correlations = []
    coefficients = []

    for _ in range(num_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=None)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            correlation, results = evaluate_fold(X_train, y_train, X_test, y_test)
            correlations.append(correlation)
            coefficients.append(results.params)

    return np.mean(correlations), np.std(correlations), pd.DataFrame(coefficients)


def analyze_imageset(data, num_repeats):
    """
    Analyze one image set
    """
    # Prepare features
    X = prepare_features(data)
    y = data["subjective_complexity"]

    # Perform cross-validation
    mean_corr, std_corr, coef_df = cross_validate_regression(X, y, num_repeats)

    # Calculate average coefficients
    avg_coefficients = coef_df.mean()

    return {
        "mean_correlation": mean_corr,
        "std_correlation": std_corr,
        "coefficients": avg_coefficients,
    }


def plot_results(data, predictions, title):
    """
    Plot actual vs predicted complexity
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data["subjective_complexity"], predictions, alpha=0.5)
    plt.plot(
        [min(data["subjective_complexity"]), max(data["subjective_complexity"])],
        [min(data["subjective_complexity"]), max(data["subjective_complexity"])],
        "r--",
    )
    plt.xlabel("Actual Subjective Complexity")
    plt.ylabel("Predicted Subjective Complexity")
    plt.title(title)
    plt.tight_layout()
    return plt


# Example usage
if __name__ == "__main__":
    # Load your data

    data = pd.read_csv("features.csv")

    # Example data structure
    data = pd.DataFrame(
        {
            "num_seg": data["# of SAM segmentations"],
            "num_class": data["# of FC-CLIP classes"],
            "subjective_complexity": data["predicted_complexity"],
            "average_similarity": data["avg_region_similarity"],
        }
    )

    # Analyze different image sets with different numbers of repeats
    # More repeats for smaller datasets
    sample_sizes = len(data)
    num_repeats = max(10, int(100 / sample_sizes))  # More repeats for smaller datasets

    results = analyze_imageset(data, num_repeats)

    print(
        f"Average Spearman Correlation: {results['mean_correlation']:.3f} ± {results['std_correlation']:.3f}"
    )
    print("\nRegression Coefficients:")
    for name, value in results["coefficients"].items():
        print(f"{name}: {value:.3f}")

    # Fit final model for visualization
    X = prepare_features(data)
    X_const = sm.add_constant(X)
    final_model = OLS(data["subjective_complexity"], X_const).fit()
    predictions = final_model.predict(X_const)

    # Plot results
    plt = plot_results(data, predictions, "Subjective Complexity: Actual vs Predicted")
    plt.show()
