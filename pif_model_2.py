import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

# Load the data
data = pd.read_csv("uvvisml-main/schmidt_2019_median_final_noduplicates_maestro.csv")
data = data.dropna()

# Remove rows with PIF(collected) greater than 100
data['PIF(collected)'] = pd.to_numeric(data['PIF(collected)'], errors='coerce')
data = data[data['median_pif'] <= 30]
print("Number of rows:", len(data))
data['median_pif'] = data['median_pif'].astype(float)
data.to_csv("filtered_data.csv", index=False)

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

# Split data into train and test sets for regression
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Split data into train and test sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Standardize features using Z-score standardization
scaler = StandardScaler()
X_train_regression_scaled = scaler.fit_transform(X_train_regression)
X_test_regression_scaled = scaler.transform(X_test_regression)

# Train regression model
regression_model = XGBRegressor()
regression_model.fit(X_train_regression_scaled, y_train_regression)

joblib.dump(regression_model, 'xgboost_regression_model.pkl')

# Predict median_pif using regression model
y_pred_regression = regression_model.predict(X_test_regression_scaled)
mse = mean_squared_error(y_test_regression, y_pred_regression)
print("Mean Squared Error for median_pif prediction:", mse)

# Train classification model
classification_model = XGBClassifier()
classification_model.fit(X_train_classification, y_train_classification)

# Predict Value using classification model
y_pred_classification = classification_model.predict(X_test_classification)
accuracy = accuracy_score(y_test_classification, y_pred_classification)
print("Accuracy for Value prediction:", accuracy)

import matplotlib.pyplot as plt

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_regression, y_pred_regression, color='blue', alpha=0.5)
plt.plot([min(y_test_regression), max(y_test_regression)], [min(y_test_regression), max(y_test_regression)], color='red')
plt.title('Actual vs. Predicted Median PIF')
plt.xlabel('Actual Median PIF')
plt.ylabel('Predicted Median PIF')
plt.show()