import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv("uvvisml-main/schmidt_2019_median_final_noduplicates_maestro.csv")
data = data.dropna()

# Convert non-numeric columns to numeric format
label_encoder = LabelEncoder()
data['Title'] = label_encoder.fit_transform(data['Title'])
data['Set'] = label_encoder.fit_transform(data['Set'])
data['SMILES'] = label_encoder.fit_transform(data['SMILES'])

# Create target variable for classification based on PIF values
data['PIF_class'] = pd.cut(data['median_pif'], bins=[float('-inf'), 3, 5, float('inf')], labels=['negative', 'uncertain', 'positive'])

# Split data into features and target variable for classification
X_classification = data.drop(columns=['median_pif', 'Value', 'PIF_class'])
y_classification = data['PIF_class']

# Split data into train and test sets for classification
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Train classification model
classification_model = XGBClassifier()
classification_model.fit(X_train_classification, y_train_classification)

# Predict PIF_class using classification model
y_pred_classification = classification_model.predict(X_test_classification)

# Evaluate classification model
accuracy = accuracy_score(y_test_classification, y_pred_classification)
print("Accuracy for PIF_class prediction:", accuracy)

# Print classification report
print(classification_report(y_test_classification, y_pred_classification))
