import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("uvvisml-main/schmidt_2019_median_final_noduplicates_maestro.csv")
data = data.dropna()

# Remove rows with PIF(collected) greater than 100
data = data[data['median_pif'] <= 30]
print("Number of rows:", len(data))

# Convert non-numeric columns to numeric format
label_encoder = LabelEncoder()
data['Title'] = label_encoder.fit_transform(data['Title'])
data['PIF(collected)'] = label_encoder.fit_transform(data['PIF(collected)'])
data['Set'] = label_encoder.fit_transform(data['Set'])
data['SMILES'] = label_encoder.fit_transform(data['SMILES'])

# Split data into features and target variables
X_regression = data.drop(columns=['median_pif', 'Value'])
y_regression = data['median_pif']
X_classification = data.drop(columns=['median_pif', 'Value'])
y_classification = data['Value']

# Standardize features using Z-score standardization
scaler = StandardScaler()
X_regression_scaled = scaler.fit_transform(X_regression)

# Define the number of folds for cross-validation
n_folds = 10

# Perform cross-validation for regression
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
r2_scores = []
for train_index, test_index in kf.split(X_regression_scaled):
    X_train, X_test = X_regression_scaled[train_index], X_regression_scaled[test_index]
    y_train, y_test = y_regression.iloc[train_index], y_regression.iloc[test_index]

    regression_model = XGBRegressor()
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)

    # Calculate R squared
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

    # Plot actual vs. predicted values for this fold
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title(f'Fold {len(r2_scores)} - Actual vs. Predicted Median PIF')
    plt.xlabel('Actual Median PIF')
    plt.ylabel('Predicted Median PIF')
    plt.show()

# Calculate average R squared
average_r2 = sum(r2_scores) / n_folds
print("Average R squared:", average_r2)