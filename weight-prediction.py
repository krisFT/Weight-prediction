import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Create synthetic data
np.random.seed(42)
height = np.random.normal(170, 10, 500)  # cm
age = np.random.normal(30, 8, 500)       # years
weight = height * 0.4 + age * 0.6 + np.random.normal(0, 5, 500)  # target

# Create a DataFrame
data = pd.DataFrame({
    'height': height,
    'age': age,
    'weight': weight
})

# Split features and target
X = data[['height', 'age']]
y = data['weight']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

# Evaluate
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Mean Absolute Error on test set: {mae:.2f} kg")

# Predict
example = np.array([[175, 25]])  # height: 175 cm, age: 25
example_scaled = scaler.transform(example)
predicted_weight = model.predict(example_scaled)
print(f"Predicted weight: {predicted_weight[0][0]:.2f} kg")

y_pred = model.predict(X_test_scaled)
plt.scatter(y_test, y_pred)
plt.xlabel("True Weight")
plt.ylabel("Predicted Weight")
plt.title("Predicted vs True Weight")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
plt.show()