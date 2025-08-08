import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("file:./mlruns")

class MLflowCallback(tf.keras.callbacks.Callback):
    """Custom callback to log training metrics to MLflow for each epoch."""
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if logs is not None:
            # Log training metrics with step parameter for time series plotting
            mlflow.log_metric(f"{self.model_name.lower()}_train_loss", logs.get('loss', 0), step=epoch)
            mlflow.log_metric(f"{self.model_name.lower()}_train_mae", logs.get('mae', 0), step=epoch)
            
            # Log validation metrics if available
            if 'val_loss' in logs:
                mlflow.log_metric(f"{self.model_name.lower()}_val_loss", logs.get('val_loss', 0), step=epoch)
            if 'val_mae' in logs:
                mlflow.log_metric(f"{self.model_name.lower()}_val_mae", logs.get('val_mae', 0), step=epoch)

def create_synthetic_data(n_samples=500, random_seed=42):
    """Create synthetic weight prediction dataset (height, age, weight)."""
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

def build_complex_dense_model(input_shape=(2,)):
    """Build a more complex dense neural network to demonstrate non-linear capabilities."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_non_linear_data(n_samples=500, random_seed=42):
    """Create synthetic data with non-linear relationships (height, age, weight)."""
    np.random.seed(random_seed)
    
    # Generate features
    height = np.random.normal(170, 10, n_samples)  # cm
    age = np.random.normal(30, 8, n_samples)       # years
    
    # Generate target with non-linear relationships
    weight = (height * 0.4 + age * 0.6 + 
              np.sin(height/10) * 5 +  # 非線性項
              np.cos(age/5) * 3 +      # 非線性項
              (height * age) / 1000 +  # 交互項
              np.random.normal(0, 5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'height': height,
        'age': age,
        'weight': weight
    })
    
    return data

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
    """Train and evaluate a model with MLflow logging."""
    # Use model-specific parameter names to avoid conflicts
    mlflow.log_param(f"{model_name.lower()}_model_type", model_name)
    mlflow.log_param(f"{model_name.lower()}_epochs", epochs)
    mlflow.log_param(f"{model_name.lower()}_input_shape", X_train.shape[1:])
    mlflow.log_param(f"{model_name.lower()}_training_samples", X_train.shape[0])
    mlflow.log_param(f"{model_name.lower()}_test_samples", X_test.shape[0])
    
    # Create custom callback for real-time tracking
    mlflow_callback = MLflowCallback(model_name)
    
    # Train with callback to track each epoch
    history = model.fit(
        X_train, y_train, 
        epochs=epochs, 
        verbose=1,  # Show progress bar
        validation_split=0.2,
        callbacks=[mlflow_callback]
    )
    
    # Log final training metrics
    mlflow.log_metric(f"{model_name.lower()}_final_train_loss", history.history['loss'][-1])
    mlflow.log_metric(f"{model_name.lower()}_final_train_mae", history.history['mae'][-1])
    if 'val_loss' in history.history:
        mlflow.log_metric(f"{model_name.lower()}_final_val_loss", history.history['val_loss'][-1])
        mlflow.log_metric(f"{model_name.lower()}_final_val_mae", history.history['val_mae'][-1])
    
    # Log training curves as artifacts
    plot_training_curves(history, model_name)
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric(f"{model_name.lower()}_test_loss", loss)
    mlflow.log_metric(f"{model_name.lower()}_test_mae", mae)
    
    print(f"[{model_name}] Mean Absolute Error on test set: {mae:.2f} kg")
    mlflow.tensorflow.log_model(model, f"{model_name.lower()}_model")
    
    return model, mae

def predict_example(model, example_data, scaler, model_name):
    """Make prediction on example data and log to MLflow."""
    if isinstance(example_data, np.ndarray):
        example_df = pd.DataFrame(example_data, columns=['height', 'age'])
    else:
        example_df = example_data
    example_scaled = scaler.transform(example_df)
    predicted_weight = model.predict(example_scaled, verbose=0)
    print(f"[{model_name}] Predicted weight: {predicted_weight[0][0]:.2f} kg")
    mlflow.log_metric(f"{model_name.lower()}_prediction", predicted_weight[0][0])
    return predicted_weight

def plot_predictions(y_true, y_pred, title, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True Weight")
    plt.ylabel("Predicted Weight")
    plt.title(title)
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = f"{model_name.lower()}_predictions.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.show()
    plt.close()

def plot_training_curves(history, model_name):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title(f'{model_name} - MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save and log to MLflow
    plot_path = f"{model_name.lower()}_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    plt.show()
    plt.close()

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
    
    print(" Original Data (Different Scales):")
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

def test_linear_vs_nonlinear():
    """Test to demonstrate linear vs non-linear model behavior."""
    print("\n" + "=" * 60)
    print("LINEAR vs NON-LINEAR DATA TEST")
    print("=" * 60)
    
    # Create linear data
    linear_data = create_synthetic_data()
    print("Linear Data (your current data):")
    print(f"Formula: weight = height * 0.4 + age * 0.6 + noise")
    print(f"Shape: {linear_data.shape}")
    
    # Create non-linear data
    non_linear_data = create_non_linear_data()
    print("\nNon-Linear Data:")
    print(f"Formula: weight = height * 0.4 + age * 0.6 + sin(height/10) * 5 + cos(age/5) * 3 + (height * age) / 1000 + noise")
    print(f"Shape: {non_linear_data.shape}")
    
    # Test simple model on linear data
    print("\n" + "-" * 40)
    print("Testing Simple Model on Linear Data:")
    X_linear, y_linear = linear_data[['height', 'age']], linear_data['weight']
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=42
    )
    
    scaler_linear = StandardScaler()
    X_train_linear_scaled = scaler_linear.fit_transform(X_train_linear)
    X_test_linear_scaled = scaler_linear.transform(X_test_linear)
    
    simple_model = build_dense_model()
    simple_model.fit(X_train_linear_scaled, y_train_linear, epochs=50, verbose=0)
    y_pred_simple = simple_model.predict(X_test_linear_scaled)
    
    # Test complex model on non-linear data
    print("\n" + "-" * 40)
    print("Testing Complex Model on Non-Linear Data:")
    X_nonlinear, y_nonlinear = non_linear_data[['height', 'age']], non_linear_data['weight']
    X_train_nonlinear, X_test_nonlinear, y_train_nonlinear, y_test_nonlinear = train_test_split(
        X_nonlinear, y_nonlinear, test_size=0.2, random_state=42
    )
    
    scaler_nonlinear = StandardScaler()
    X_train_nonlinear_scaled = scaler_nonlinear.fit_transform(X_train_nonlinear)
    X_test_nonlinear_scaled = scaler_nonlinear.transform(X_test_nonlinear)
    
    complex_model = build_complex_dense_model()
    complex_model.fit(X_train_nonlinear_scaled, y_train_nonlinear, epochs=50, verbose=0)
    y_pred_complex = complex_model.predict(X_test_nonlinear_scaled)
    
    # Also test simple model on non-linear data for comparison
    simple_model_nonlinear = build_dense_model()
    simple_model_nonlinear.fit(X_train_nonlinear_scaled, y_train_nonlinear, epochs=50, verbose=0)
    y_pred_simple_nonlinear = simple_model_nonlinear.predict(X_test_nonlinear_scaled)
    
    # Calculate MAE for comparison
    mae_simple_linear = mean_absolute_error(y_test_linear, y_pred_simple)
    mae_complex_nonlinear = mean_absolute_error(y_test_nonlinear, y_pred_complex)
    mae_simple_nonlinear = mean_absolute_error(y_test_nonlinear, y_pred_simple_nonlinear)
    
    print(f"\nMAE Comparison:")
    print(f"Simple Model on Linear Data: {mae_simple_linear:.2f}")
    print(f"Complex Model on Non-Linear Data: {mae_complex_nonlinear:.2f}")
    print(f"Simple Model on Non-Linear Data: {mae_simple_nonlinear:.2f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Linear data with simple model
    ax1.scatter(y_test_linear, y_pred_simple, alpha=0.6)
    ax1.plot([y_test_linear.min(), y_test_linear.max()], [y_test_linear.min(), y_test_linear.max()], 'r--', label='Perfect Prediction')
    ax1.set_title(f'Simple Model on Linear Data\nMAE: {mae_simple_linear:.2f}')
    ax1.set_xlabel('True Weight')
    ax1.set_ylabel('Predicted Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Non-linear data with complex model
    ax2.scatter(y_test_nonlinear, y_pred_complex, alpha=0.6)
    ax2.plot([y_test_nonlinear.min(), y_test_nonlinear.max()], [y_test_nonlinear.min(), y_test_nonlinear.max()], 'r--', label='Perfect Prediction')
    ax2.set_title(f'Complex Model on Non-Linear Data\nMAE: {mae_complex_nonlinear:.2f}')
    ax2.set_xlabel('True Weight')
    ax2.set_ylabel('Predicted Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Non-linear data with simple model (for comparison)
    ax3.scatter(y_test_nonlinear, y_pred_simple_nonlinear, alpha=0.6)
    ax3.plot([y_test_nonlinear.min(), y_test_nonlinear.max()], [y_test_nonlinear.min(), y_test_nonlinear.max()], 'r--', label='Perfect Prediction')
    ax3.set_title(f'Simple Model on Non-Linear Data\nMAE: {mae_simple_nonlinear:.2f}')
    ax3.set_xlabel('True Weight')
    ax3.set_ylabel('Predicted Weight')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Feature relationships comparison
    ax4.scatter(linear_data['height'], linear_data['weight'], alpha=0.6, label='Linear Data')
    ax4.scatter(non_linear_data['height'], non_linear_data['weight'], alpha=0.6, label='Non-Linear Data')
    ax4.set_title('Data Comparison: Height vs Weight')
    ax4.set_xlabel('Height')
    ax4.set_ylabel('Weight')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_vs_nonlinear_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print("\n" + "=" * 60)
    print(f"Simple model on non-linear data MAE: {mae_simple_nonlinear:.2f}")
    print(f"Complex model on non-linear data MAE: {mae_complex_nonlinear:.2f}")
    print("=" * 60)

def compare_model_architectures():
    """Compare different model architectures and explain their performance."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    # Create sample data
    data = create_synthetic_data()
    X, y = data[['height', 'age']], data['weight']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different models
    models = {
        'Simple Dense': build_dense_model(),
        'Complex Dense': build_complex_dense_model(),
        '1D CNN': build_cnn_model()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train, epochs=50, verbose=0)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = mae
        print(f"{name} MAE: {mae:.2f}")
    
    # Print comparison
    print("\n" + "=" * 40)
    print("PERFORMANCE COMPARISON")
    print("=" * 40)
    for name, mae in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name}: {mae:.2f}")
    
    # Model complexity comparison
    print("\n" + "=" * 40)
    print("MODEL COMPLEXITY")
    print("=" * 40)
    
    simple_dense = build_dense_model()
    complex_dense = build_complex_dense_model()
    cnn = build_cnn_model()
    
    print(f"Simple Dense parameters: {simple_dense.count_params():,}")
    print(f"Complex Dense parameters: {complex_dense.count_params():,}")
    print(f"1D CNN parameters: {cnn.count_params():,}")


def create_synthetic_data_3features(n_samples=500, random_seed=42):
    """Create synthetic weight prediction dataset with 3 features (height, age, activity_level, weight)."""
    np.random.seed(random_seed)
    
    # Generate features
    height = np.random.normal(170, 10, n_samples)  # cm
    age = np.random.normal(30, 8, n_samples)       # years
    activity_level = np.random.choice([1, 2, 3, 4, 5], n_samples)  # 1=sedentary, 5=very active
    
    # Generate target (weight) with more complex relationships
    weight = (height * 0.4 + age * 0.6 + 
              activity_level * (-2) +  # More active = lower weight
              (height * age) / 1000 +  # Interaction term
              np.random.normal(0, 5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'height': height,
        'age': age,
        'activity_level': activity_level,
        'weight': weight
    })
    
    return data

def create_non_linear_data_3features(n_samples=500, random_seed=42):
    """Create synthetic data with 3 features and non-linear relationships (height, age, activity_level, weight)."""
    np.random.seed(random_seed)
    
    # Generate features
    height = np.random.normal(170, 10, n_samples)  # cm
    age = np.random.normal(30, 8, n_samples)       # years
    activity_level = np.random.choice([1, 2, 3, 4, 5], n_samples)  # 1=sedentary, 5=very active
    
    # Generate target with complex non-linear relationships
    weight = (height * 0.4 + age * 0.6 + 
              activity_level * (-2) +
              np.sin(height/10) * 5 +  # Non-linear term
              np.cos(age/5) * 3 +      # Non-linear term
              np.exp(-activity_level/2) * 10 +  # Exponential relationship
              (height * age * activity_level) / 10000 +  # 3-way interaction
              np.random.normal(0, 5, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'height': height,
        'age': age,
        'activity_level': activity_level,
        'weight': weight
    })
    
    return data

def prepare_data_3features(data, test_size=0.2, random_state=42):
    """Prepare 3-feature data for training by splitting and scaling."""
    # Split features and target
    X = data[['height', 'age', 'activity_level']]
    y = data['weight']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def build_dense_model_3features(input_shape=(3,)):
    """Build a dense neural network for 3 features."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_complex_dense_model_3features(input_shape=(3,)):
    """Build a more complex dense neural network for 3 features."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_model_3features(input_shape=(3, 1)):
    """Build a 1D CNN model for 3 features."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def test_3features_comparison():
    """Test different models on 3-feature dataset."""
    print("\n" + "=" * 60)
    print("3-FEATURE DATASET COMPARISON")
    print("=" * 60)
    
    # Create 3-feature data
    linear_data_3f = create_synthetic_data_3features()
    non_linear_data_3f = create_non_linear_data_3features()
    
    print("Linear 3-Feature Data:")
    print(f"Features: {list(linear_data_3f.columns[:-1])}")
    print(f"Formula: weight = height*0.4 + age*0.6 + activity*(-2) + (height*age)/1000 + noise")
    print(f"Shape: {linear_data_3f.shape}")
    
    print("\nNon-Linear 3-Feature Data:")
    print(f"Features: {list(non_linear_data_3f.columns[:-1])}")
    print(f"Formula: weight = height*0.4 + age*0.6 + activity*(-2) + sin(height/10)*5 + cos(age/5)*3 + exp(-activity/2)*10 + (height*age*activity)/10000 + noise")
    print(f"Shape: {non_linear_data_3f.shape}")
    
    # Test on linear data
    print("\n" + "-" * 40)
    print("Testing on Linear 3-Feature Data:")
    X_linear, y_linear = linear_data_3f[['height', 'age', 'activity_level']], linear_data_3f['weight']
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear, y_linear, test_size=0.2, random_state=42
    )
    
    scaler_linear = StandardScaler()
    X_train_linear_scaled = scaler_linear.fit_transform(X_train_linear)
    X_test_linear_scaled = scaler_linear.transform(X_test_linear)
    
    # Test different models
    models_linear = {
        'Simple Dense': build_dense_model_3features(),
        'Complex Dense': build_complex_dense_model_3features(),
        '1D CNN': build_cnn_model_3features()
    }
    
    results_linear = {}
    predictions_linear = {}
    
    for name, model in models_linear.items():
        print(f"\nTraining {name} on linear 3-Feature data...")
        history = model.fit(X_train_linear_scaled, y_train_linear, epochs=50, verbose=0, validation_split=0.2)
        y_pred = model.predict(X_test_linear_scaled)
        mae = mean_absolute_error(y_test_linear, y_pred)
        results_linear[name] = mae
        predictions_linear[name] = y_pred
        print(f"{name} MAE: {mae:.2f}")
        
        # Plot training curves for 3-feature models
        plot_training_curves_3features(history, name, "Linear")
        
        # Plot predictions for 3-feature models
        plot_predictions_3features(y_test_linear, y_pred, f"{name} on Linear 3-Feature Data", name, "Linear")
    
    # Test on non-linear data
    print("\n" + "-" * 40)
    print("Testing on Non-Linear 3-Feature Data:")
    X_nonlinear, y_nonlinear = non_linear_data_3f[['height', 'age', 'activity_level']], non_linear_data_3f['weight']
    X_train_nonlinear, X_test_nonlinear, y_train_nonlinear, y_test_nonlinear = train_test_split(
        X_nonlinear, y_nonlinear, test_size=0.2, random_state=42
    )
    
    scaler_nonlinear = StandardScaler()
    X_train_nonlinear_scaled = scaler_nonlinear.fit_transform(X_train_nonlinear)
    X_test_nonlinear_scaled = scaler_nonlinear.transform(X_test_nonlinear)
    
    models_nonlinear = {
        'Simple Dense': build_dense_model_3features(),
        'Complex Dense': build_complex_dense_model_3features(),
        '1D CNN': build_cnn_model_3features()
    }
    
    results_nonlinear = {}
    predictions_nonlinear = {}
    
    for name, model in models_nonlinear.items():
        print(f"\nTraining {name} on non-linear 3-Feature data...")
        history = model.fit(X_train_nonlinear_scaled, y_train_nonlinear, epochs=50, verbose=0, validation_split=0.2)
        y_pred = model.predict(X_test_nonlinear_scaled)
        mae = mean_absolute_error(y_test_nonlinear, y_pred)
        results_nonlinear[name] = mae
        predictions_nonlinear[name] = y_pred
        print(f"{name} MAE: {mae:.2f}")
        
        # Plot training curves for 3-feature models
        plot_training_curves_3features(history, name, "NonLinear")
        
        # Plot predictions for 3-feature models
        plot_predictions_3features(y_test_nonlinear, y_pred, f"{name} on Non-Linear 3-Feature Data", name, "NonLinear")
    
    # Create comprehensive comparison plots
    plot_3features_comprehensive_comparison(results_linear, results_nonlinear, predictions_linear, predictions_nonlinear, 
                                          y_test_linear, y_test_nonlinear)
    
    # Print comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON - LINEAR DATA")
    print("=" * 50)
    for name, mae in sorted(results_linear.items(), key=lambda x: x[1]):
        print(f"{name}: {mae:.2f}")
    
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON - NON-LINEAR DATA")
    print("=" * 50)
    for name, mae in sorted(results_nonlinear.items(), key=lambda x: x[1]):
        print(f"{name}: {mae:.2f}")
    
    # Model complexity comparison
    print("\n" + "=" * 50)
    print("MODEL COMPLEXITY (3 FEATURES)")
    print("=" * 50)
    
    simple_dense_3f = build_dense_model_3features()
    complex_dense_3f = build_complex_dense_model_3features()
    cnn_3f = build_cnn_model_3features()
    
    print(f"Simple Dense parameters: {simple_dense_3f.count_params():,}")
    print(f"Complex Dense parameters: {complex_dense_3f.count_params():,}")
    print(f"1D CNN parameters: {cnn_3f.count_params():,}")
    
    print("\n" + "=" * 50)
    print("CONCLUSION - 3 FEATURES")
    print("=" * 50)
    print("With 3 features:")
    print("- More complex interactions possible")
    print("- CNN may show more advantages")
    print("- Complex models have more room to learn")
    print("- Activity level adds meaningful variation")

def plot_training_curves_3features(history, model_name, data_type):
    """Plot and save training curves for 3-feature models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss ({data_type} Data)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    if 'val_mae' in history.history:
        ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title(f'{model_name} - MAE ({data_type} Data)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save and log to MLflow
    plot_path = f"{model_name.lower().replace(' ', '_')}_{data_type.lower()}_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    plt.show()
    plt.close()

def plot_predictions_3features(y_true, y_pred, title, model_name, data_type):
    """Plot predictions for 3-feature models."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("True Weight")
    plt.ylabel("Predicted Weight")
    plt.title(title)
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = f"{model_name.lower().replace(' ', '_')}_{data_type.lower()}_predictions.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.show()
    plt.close()

def plot_3features_comprehensive_comparison(results_linear, results_nonlinear, predictions_linear, predictions_nonlinear, 
                                          y_test_linear, y_test_nonlinear):
    """Create comprehensive comparison plots for 3-feature models."""
    
    # Performance comparison bar plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Bar plot for linear data
    models = list(results_linear.keys())
    mae_values_linear = list(results_linear.values())
    bars1 = ax1.bar(models, mae_values_linear, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('MAE Comparison - Linear 3-Feature Data')
    ax1.set_ylabel('MAE')
    ax1.set_ylim(0, max(mae_values_linear) * 1.2)
    for bar, value in zip(bars1, mae_values_linear):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}', 
                ha='center', va='bottom')
    
    # Bar plot for non-linear data
    mae_values_nonlinear = list(results_nonlinear.values())
    bars2 = ax2.bar(models, mae_values_nonlinear, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('MAE Comparison - Non-Linear 3-Feature Data')
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0, max(mae_values_nonlinear) * 1.2)
    for bar, value in zip(bars2, mae_values_nonlinear):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}', 
                ha='center', va='bottom')
    
    # Prediction scatter plots for linear data
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    for i, (name, y_pred) in enumerate(predictions_linear.items()):
        ax3.scatter(y_test_linear, y_pred, alpha=0.6, color=colors[i], label=f'{name} (MAE: {results_linear[name]:.2f})')
    ax3.plot([y_test_linear.min(), y_test_linear.max()], [y_test_linear.min(), y_test_linear.max()], 'r--', label='Perfect Prediction')
    ax3.set_title('Predictions - Linear 3-Feature Data')
    ax3.set_xlabel('True Weight')
    ax3.set_ylabel('Predicted Weight')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Prediction scatter plots for non-linear data
    for i, (name, y_pred) in enumerate(predictions_nonlinear.items()):
        ax4.scatter(y_test_nonlinear, y_pred, alpha=0.6, color=colors[i], label=f'{name} (MAE: {results_nonlinear[name]:.2f})')
    ax4.plot([y_test_nonlinear.min(), y_test_nonlinear.max()], [y_test_nonlinear.min(), y_test_nonlinear.max()], 'r--', label='Perfect Prediction')
    ax4.set_title('Predictions - Non-Linear 3-Feature Data')
    ax4.set_xlabel('True Weight')
    ax4.set_ylabel('Predicted Weight')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "3features_comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    plt.show()
    plt.close()
    
    # Model complexity comparison
    simple_dense_3f = build_dense_model_3features()
    complex_dense_3f = build_complex_dense_model_3features()
    cnn_3f = build_cnn_model_3features()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    models_complexity = ['Simple Dense', 'Complex Dense', '1D CNN']
    params = [simple_dense_3f.count_params(), complex_dense_3f.count_params(), cnn_3f.count_params()]
    
    bars = ax.bar(models_complexity, params, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title('Model Complexity Comparison (3 Features)')
    ax.set_ylabel('Number of Parameters')
    for bar, param in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01, f'{param:,}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = "3features_model_complexity.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow.log_artifact(plot_path)
    plt.show()
    plt.close()

def main():
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_experiment("weight_prediction")
    with mlflow.start_run(run_name=f"weight_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print("=" * 60)
        print("WEIGHT PREDICTION: DENSE vs CONV1D COMPARISON WITH MLFLOW")
        print("=" * 60)
        
        # Add the linear vs non-linear test
        test_linear_vs_nonlinear()
        
        # Add model architecture comparison
        compare_model_architectures()
        
        # Add 3-feature dataset comparison
        test_3features_comparison()
        
        mlflow.log_param("data_samples", 500)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_seed", 42)
        data = create_synthetic_data()
        mlflow.log_param("dataset_shape", data.shape)
        mlflow.log_param("features", list(data.columns[:-1]))
        mlflow.log_param("target", data.columns[-1])
        print("\n Dataset Overview:")
        print(data.head())
        print(f"\nDataset shape: {data.shape}")
        print(f"Features: {list(data.columns[:-1])}")
        print(f"Target: {data.columns[-1]}")
        demonstrate_scaling()
        print("\n" + "=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)
        X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(data)
        mlflow.log_param("training_set_size", X_train_scaled.shape[0])
        mlflow.log_param("test_set_size", X_test_scaled.shape[0])
        mlflow.log_param("feature_count", X_train_scaled.shape[1])
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print("\n" + "=" * 60)
        print("DENSE NEURAL NETWORK MODEL")
        print("=" * 60)
        dense_params = {
            "dense_hidden_units": [16, 8],
            "dense_activation": "relu",
            "dense_optimizer": "adam",
            "dense_loss": "mse"
        }
        mlflow.log_params(dense_params)
        dense_model = build_dense_model()
        dense_model, dense_mae = train_and_evaluate_model(
            dense_model, X_train_scaled, y_train, X_test_scaled, y_test, "Dense"
        )
        example = np.array([[175, 25]])
        dense_prediction = predict_example(dense_model, example, scaler, "Dense")
        y_pred_dense = dense_model.predict(X_test_scaled, verbose=0)
        plot_predictions(y_test, y_pred_dense, "Dense Neural Network: Predicted vs True Weight", "Dense")
        print("\n" + "=" * 60)
        print("COMPLEX DENSE NEURAL NETWORK MODEL")
        print("=" * 60)
        complex_dense_params = {
            "complex_dense_hidden_units": [64, 32, 16, 8],
            "complex_dense_dropout_rate": [0.3, 0.2],
            "complex_dense_activation": "relu",
            "complex_dense_optimizer": "adam",
            "complex_dense_loss": "mse"
        }
        mlflow.log_params(complex_dense_params)
        complex_dense_model = build_complex_dense_model()
        complex_dense_model, complex_dense_mae = train_and_evaluate_model(
            complex_dense_model, X_train_scaled, y_train, X_test_scaled, y_test, "Complex Dense"
        )
        example_complex = np.array([[175, 25]])
        complex_dense_prediction = predict_example(complex_dense_model, example_complex, scaler, "Complex Dense")
        y_pred_complex_dense = complex_dense_model.predict(X_test_scaled, verbose=0)
        plot_predictions(y_test, y_pred_complex_dense, "Complex Dense Neural Network: Predicted vs True Weight", "Complex Dense")
        print("\n" + "=" * 60)
        print("1D CONVOLUTIONAL NEURAL NETWORK MODEL")
        print("=" * 60)
        cnn_params = {
            "cnn_filters": 32,
            "cnn_kernel_size": 2,
            "cnn_activation": "relu",
            "cnn_optimizer": "adam",
            "cnn_loss": "mse"
        }
        mlflow.log_params(cnn_params)
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        cnn_model = build_cnn_model()
        cnn_model, cnn_mae = train_and_evaluate_model(
            cnn_model, X_train_cnn, y_train, X_test_cnn, y_test, "1D CNN"
        )
        example_cnn = example.reshape(1, 2, 1)
        example_df = pd.DataFrame(example, columns=['height', 'age'])
        example_scaled_cnn = scaler.transform(example_df).reshape(1, 2, 1)
        cnn_prediction = cnn_model.predict(example_scaled_cnn, verbose=0)
        print(f"[1D CNN] Predicted weight: {cnn_prediction[0][0]:.2f} kg")
        y_pred_cnn = cnn_model.predict(X_test_cnn, verbose=0)
        plot_predictions(y_test, y_pred_cnn, "1D CNN: Predicted vs True Weight", "CNN")
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"Dense Neural Network MAE: {dense_mae:.2f} kg")
        print(f"Complex Dense Neural Network MAE: {complex_dense_mae:.2f} kg")
        print(f"1D CNN MAE: {cnn_mae:.2f} kg")
        mlflow.log_metric("dense_mae", dense_mae)
        mlflow.log_metric("complex_dense_mae", complex_dense_mae)
        mlflow.log_metric("cnn_mae", cnn_mae)
        mlflow.log_metric("mae_difference_dense_complex", abs(dense_mae - complex_dense_mae))
        mlflow.log_metric("mae_difference_dense_cnn", abs(dense_mae - cnn_mae))
        mlflow.log_metric("mae_difference_complex_cnn", abs(complex_dense_mae - cnn_mae))
        if dense_mae < complex_dense_mae and dense_mae < cnn_mae:
            best_model = "Dense"
            improvement = min(complex_dense_mae, cnn_mae) - dense_mae
            print(f"Dense model performs better by {improvement:.2f} kg")
        elif complex_dense_mae < dense_mae and complex_dense_mae < cnn_mae:
            best_model = "Complex Dense"
            improvement = min(dense_mae, cnn_mae) - complex_dense_mae
            print(f"Complex Dense model performs better by {improvement:.2f} kg")
        elif cnn_mae < dense_mae and cnn_mae < complex_dense_mae:
            best_model = "CNN"
            improvement = min(dense_mae, complex_dense_mae) - cnn_mae
            print(f"CNN model performs better by {improvement:.2f} kg")
        else:
            best_model = "Equal"
            print("All models perform equally well")
        mlflow.log_param("best_model", best_model)
        if best_model != "Equal":
            mlflow.log_metric("improvement", improvement)
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE - Check MLflow UI for detailed results!")
        print("=" * 60)
        print(f"\n Best Model: {best_model}")
        print(f" Dense MAE: {dense_mae:.2f} kg")
        print(f" Complex Dense MAE: {complex_dense_mae:.2f} kg")
        print(f" CNN MAE: {cnn_mae:.2f} kg")
 

if __name__ == "__main__":
    main()