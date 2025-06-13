import pandas as pd

def select_features_by_correlation(correlation_series, threshold=0.1):
    selected = correlation_series[correlation_series > threshold].index.tolist()
    if 'CTC' in selected:
        selected.remove('CTC')
    return selected

# Load precomputed correlations
placement_corr = pd.read_pickle("placement_correlation.pkl")

# Select features using threshold
selected_features = select_features_by_correlation(placement_corr, threshold=0.1)

print("Selected features:", selected_features)

pd.Series(selected_features).to_pickle("selected_features.pkl")