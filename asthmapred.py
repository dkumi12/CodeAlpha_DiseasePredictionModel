# Clean y by dropping NaN values and align X accordingly

import pandas as pd
import numpy as np

# Read the enhanced dataset
enhanced_df = pd.read_csv('Enhanced_AsthmaDiseasePrediction.csv')

# Drop rows where Asthma_Severity is Unknown
enhanced_df = enhanced_df[enhanced_df['Asthma_Severity'] != 'Unknown']

# Drop the one-hot encoded columns
cols_to_drop = ['Severity_Mild', 'Severity_Moderate', 'Severity_None', 
                'Age_0-9', 'Age_10-19', 'Age_20-24', 'Age_25-59', 'Age_60+',
                'Gender_Female', 'Gender_Male']

df_model = enhanced_df.drop(columns=cols_to_drop)

# Separate target and predictors
target = 'Asthma_Severity'
X = df_model.drop(columns=[target])
y = df_model[target]

# Drop any rows with NaN in y (target)
y_clean = y.dropna()

# Align X with y_clean by selecting only indices available in y_clean
X_clean = X.loc[y_clean.index]

# Verify NaN in y_clean and X_clean
def report_nans():
    print('NaN count in X_clean:', X_clean.isna().sum().sum())
    print('NaN count in y_clean:', y_clean.isna().sum())

report_nans()

# Check unique values in y_clean
print('Unique values in Asthma_Severity:', y_clean.unique())

# Now proceed with train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)
print('Train-test split successful')

# Continue with modeling
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Identify numeric and categorical features
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print('Numeric features:', numeric_features)
print('Categorical features:', categorical_features)

# Create a preprocessor pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Define a pipeline with logistic regression classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit the model
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Classification Report:\
", report)
print("Accuracy:", accuracy)

# Save the model pipeline using joblib
joblib.dump(clf, 'asthma_severity_model.joblib')
print("Model saved as asthma_severity_model.joblib")
print("Download the model from: https://julius.ai/files?filename=asthma_severity_model.joblib")

print("Prediction model training completed")