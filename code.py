import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your dataset with historical data (e.g., containing socio-economic status, age, gender, and survival labels)
# Replace 'your_dataset.csv' with your actual dataset file.
data = pd.read_csv('your_dataset.csv')

# Data Preprocessing
X = data[['SocioEconomicStatus', 'Age', 'Gender']]
y = data['Survival']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a Gradient Boosting Classifier
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.05, 0.01]
}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Find the best hyperparameters
best_clf = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5)

# Train the best model
best_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Now you can use this model to predict survival
new_data = pd.DataFrame({'SocioEconomicStatus': [1], 'Age': [25], 'Gender': [0]})
new_data = scaler.transform(new_data)
prediction = best_clf.predict(new_data)
if prediction[0] == 1:
    print("The person is likely to be safe from sinking.")
else:
    print("The person may be at risk of sinking.")
