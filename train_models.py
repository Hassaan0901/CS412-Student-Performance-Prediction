"""CS412: train and compare four GPA regression models."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the same dataset used for the exploration section.
df = pd.read_csv('student_lifestyle_dataset.csv')

# GPA is the value to predict. Student_ID is only an identifier.
X = df.drop(columns=['Student_ID', 'GPA'])
X = pd.get_dummies(X, columns=['Stress_Level'], dtype=int)
y = df['GPA']

# All four models use the same 80/20 train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Only the neural network needs scaling in this experiment.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
}

results = []
for name, model in models.items():
    train_inputs, test_inputs = (X_train_scaled, X_test_scaled) if name == 'Neural Network' else (X_train, X_test)
    model.fit(train_inputs, y_train)
    predictions = model.predict(test_inputs)
    results.append({
        'Model': name,
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'R2': r2_score(y_test, predictions),
    })

results_df = pd.DataFrame(results)
print('Training records:', len(X_train))
print('Testing records:', len(X_test))
print('\nModel comparison:')
print(results_df.to_string(index=False, formatters={c: '{:.4f}'.format for c in ['MAE', 'RMSE', 'R2']}))
results_df.to_csv('model_results.csv', index=False)
print('\nSaved results to model_results.csv')
