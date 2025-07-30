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

def main():
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_experiment("weight_prediction")
    with mlflow.start_run(run_name=f"weight_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print("=" * 60)
        print("WEIGHT PREDICTION: DENSE vs CONV1D COMPARISON WITH MLFLOW")
        print("=" * 60)
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
            "hidden_units": [16, 8],
            "activation": "relu",
            "optimizer": "adam",
            "loss": "mse"
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
        print("1D CONVOLUTIONAL NEURAL NETWORK MODEL")
        print("=" * 60)
        cnn_params = {
            "filters": 32,
            "kernel_size": 2,
            "activation": "relu",
            "optimizer": "adam",
            "loss": "mse"
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
        print(f"1D CNN MAE: {cnn_mae:.2f} kg")
        mlflow.log_metric("dense_mae", dense_mae)
        mlflow.log_metric("cnn_mae", cnn_mae)
        mlflow.log_metric("mae_difference", abs(dense_mae - cnn_mae))
        if dense_mae < cnn_mae:
            best_model = "Dense"
            improvement = cnn_mae - dense_mae
            print(f"Dense model performs better by {improvement:.2f} kg")
        elif cnn_mae < dense_mae:
            best_model = "CNN"
            improvement = dense_mae - cnn_mae
            print(f"CNN model performs better by {improvement:.2f} kg")
        else:
            best_model = "Equal"
            print("Both models perform equally well")
        mlflow.log_param("best_model", best_model)
        if best_model != "Equal":
            mlflow.log_metric("improvement", improvement)
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE - Check MLflow UI for detailed results!")
        print("=" * 60)
        print(f"\n Best Model: {best_model}")
        print(f" Dense MAE: {dense_mae:.2f} kg")
        print(f" CNN MAE: {cnn_mae:.2f} kg")
 

if __name__ == "__main__":
    main()