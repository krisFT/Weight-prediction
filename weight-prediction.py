import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def create_synthetic_data(n_samples=500, random_seed=42):
    """Create synthetic weight prediction dataset."""
    np.random.seed(random_seed)
    
    # Generate features
    height = np.random.normal(170, 10, n_samples)  # cm
    age = np.random.normal(30, 8, n_samples)       # years
    
    # Generate target (weight) with some relationship to features
    weight = height * 0.4 + age * 0.6 + np.random.normal(0, 5, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'height': height,
        'age': age,
        'weight': weight
    })
    
    return data

def prepare_data(data, test_size=0.2, random_state=42):
    """Prepare data for training by splitting and scaling."""
    # Split features and target
    X = data[['height', 'age']]
    y = data['weight']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features using StandardScaler
    # Why StandardScaler?
    # 1. Height values: 150-190 cm (large scale)
    # 2. Age values: 20-40 years (smaller scale)
    # 3. Without scaling, height would dominate the model
    # 4. StandardScaler makes both features have mean=0, std=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
    X_test_scaled = scaler.transform(X_test)        # Apply same scaling to test data
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_dense_model(input_shape=(2,)):
    """Build a simple dense neural network."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1) #Binary: Dense(1, activation='sigmoid') #more classes: Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_model(input_shape=(2, 1)):
    """Build a 1D CNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, epochs=100):
    """Train and evaluate a model."""
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    # plt.plot(model.history['loss'])
    # plt.plot(model.history['mae'])
    # plt.show()

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{model_name}] Mean Absolute Error on test set: {mae:.2f} kg")
    
    return model, mae

def predict_example(model, example_data, scaler, model_name):
    """Make prediction on example data."""
    # Convert to DataFrame to preserve feature names
    if isinstance(example_data, np.ndarray):
        example_df = pd.DataFrame(example_data, columns=['height', 'age'])
    else:
        example_df = example_data
    
    example_scaled = scaler.transform(example_df)
    predicted_weight = model.predict(example_scaled, verbose=0)
    print(f"[{model_name}] Predicted weight: {predicted_weight[0][0]:.2f} kg")
    return predicted_weight

def plot_predictions(y_true, y_pred, title):
    """Plot predicted vs true values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True Weight")
    plt.ylabel("Predicted Weight")
    plt.title(title)
    
    # Add reference line
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def demonstrate_scaling():
    """Demonstrate why StandardScaler is important."""
    print("\n" + "=" * 60)
    print("WHY DO WE NEED STANDARDSCALER?")
    print("=" * 60)
    
    # Create sample data with different scales
    sample_data = pd.DataFrame({
        'height': [150, 160, 170, 180, 190],  # cm - values around 150-190
        'age': [20, 25, 30, 35, 40],          # years - values around 20-40
        'weight': [50, 60, 70, 80, 90]        # kg - target variable
    })
    
    print("📊 Original Data (Different Scales):")
    print(sample_data)
    print(f"\nHeight range: {sample_data['height'].min()} - {sample_data['height'].max()} cm")
    print(f"Age range: {sample_data['age'].min()} - {sample_data['age'].max()} years")
    print(f"Weight range: {sample_data['weight'].min()} - {sample_data['weight'].max()} kg")
    
    # Apply StandardScaler
    scaler_demo = StandardScaler()
    features = sample_data[['height', 'age']]
    scaled_features = scaler_demo.fit_transform(features)
    
    print("\n After StandardScaler:")
    scaled_df = pd.DataFrame(scaled_features, columns=['height_scaled', 'age_scaled'])
    print(scaled_df)
    print(f"\nHeight scaled range: {scaled_df['height_scaled'].min():.2f} - {scaled_df['height_scaled'].max():.2f}")
    print(f"Age scaled range: {scaled_df['age_scaled'].min():.2f} - {scaled_df['age_scaled'].max():.2f}")
    
    print("\n Benefits of StandardScaler:")
    print("1. All features now have similar scales (around -1 to +1)")
    print("2. Neural networks train faster and more reliably")
    print("3. Prevents features with larger values from dominating")
    print("4. Helps gradient descent converge better")

def main():
    """Main function to run the weight prediction comparison."""
    
    # ============================================================================
    # 1. Data Creation and Exploration
    # ============================================================================
    print("=" * 60)
    print("WEIGHT PREDICTION: DENSE vs CONV1D COMPARISON")
    print("=" * 60)
    
    # Create synthetic data
    data = create_synthetic_data()
    print("\n📊 Dataset Overview:")
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    print(f"Features: {list(data.columns[:-1])}") #data.columns	Index([...]) 欄位名稱 #data.iloc[0]	第 1 列的所有值	第 1 row 的數據 #data.iloc[:, 0]	第一欄的所有值	第 1 column 的數據
    print(f"Target: {data.columns[-1]}")
    
    # Demonstrate scaling
    demonstrate_scaling()
    
    # ============================================================================
    # 2. Data Preparation
    # ============================================================================
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data)
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    
    # ============================================================================
    # 3. Dense Neural Network Model
    # ============================================================================
    print("\n" + "=" * 60)
    print("DENSE NEURAL NETWORK MODEL")
    print("=" * 60)
    
    # Build and train dense model
    dense_model = build_dense_model()
    dense_model, dense_mae = train_and_evaluate_model(
        dense_model, X_train_scaled, y_train, X_test_scaled, y_test, "Dense"
    )
    
    # Make predictions
    example = np.array([[175, 25]])  # height: 175 cm, age: 25
    dense_prediction = predict_example(dense_model, example, scaler, "Dense")
    
    # Plot dense model predictions
    y_pred_dense = dense_model.predict(X_test_scaled, verbose=0)
    plot_predictions(y_test, y_pred_dense, "Dense Neural Network: Predicted vs True Weight")
    
    # ============================================================================
    # 4. 1D CNN Model
    # ============================================================================
    print("\n" + "=" * 60)
    print("1D CONVOLUTIONAL NEURAL NETWORK MODEL")
    print("=" * 60)
    
    # Reshape data for CNN: (samples, timesteps, features)
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1) 
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Build and train CNN model
    cnn_model = build_cnn_model()
    cnn_model, cnn_mae = train_and_evaluate_model(
        cnn_model, X_train_cnn, y_train, X_test_cnn, y_test, "1D CNN"
    )
    
    # Make predictions with CNN
    example_cnn = example.reshape(1, 2, 1)  # (samples=1, timesteps=2, features=1)
    # Create DataFrame with proper feature names for scaling
    example_df = pd.DataFrame(example, columns=['height', 'age'])
    example_scaled_cnn = scaler.transform(example_df).reshape(1, 2, 1)
    cnn_prediction = cnn_model.predict(example_scaled_cnn, verbose=0)
    print(f"[1D CNN] Predicted weight: {cnn_prediction[0][0]:.2f} kg")
    
    # Plot CNN model predictions
    y_pred_cnn = cnn_model.predict(X_test_cnn, verbose=0)
    plot_predictions(y_test, y_pred_cnn, "1D CNN: Predicted vs True Weight")
    
    # ============================================================================
    # 5. Model Comparison
    # ============================================================================
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    print(f"Dense Neural Network MAE: {dense_mae:.2f} kg")
    print(f"1D CNN MAE: {cnn_mae:.2f} kg")
    
    if dense_mae < cnn_mae:
        print(f"Dense model performs better by {cnn_mae - dense_mae:.2f} kg")
    elif cnn_mae < dense_mae:
        print(f"CNN model performs better by {dense_mae - cnn_mae:.2f} kg")
    else:
        print("🤝 Both models perform equally well")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()