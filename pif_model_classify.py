import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("uvvisml-main/schmidt_2019_median_final_noduplicates_maestro.csv")
data = data.dropna()

label_encoder = LabelEncoder()
data['Title'] = label_encoder.fit_transform(data['Title'])
data['PIF(collected)'] = label_encoder.fit_transform(data['PIF(collected)'])
data['Set'] = label_encoder.fit_transform(data['Set'])
data['SMILES'] = label_encoder.fit_transform(data['SMILES'])

data['PIF(collected)'] = pd.to_numeric(data['PIF(collected)'], errors='coerce')
data = data[data['median_pif'] <= 100]
print("Number of rows:", len(data))
data['median_pif'] = data['median_pif'].astype(float)
data.to_csv("filtered_data.csv", index=False)

X = data.drop(columns=['median_pif', 'Value'])
y = data['median_pif']

# Convert median_pif values into classes
def classify_median_pif(value):
    if value < 3:
        return 'Negative'
    elif value > 5:
        return 'Positive'
    else:
        return 'Uncertain'

# Convert median_pif values to classes
y_class = np.array([classify_median_pif(value) for value in y])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Data preprocessing (optional)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a classification model
classifier = RandomForestClassifier()
classifier.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_scaled)
print(classification_report(y_test, y_pred))