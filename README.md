# Weight Prediction: Dense vs 1D CNN with MLflow

Predict a person's weight from simple features (height, age, and an optional activity level) using TensorFlow. This project generates synthetic datasets, trains multiple neural network architectures, compares their performance, and logs everything to MLflow for easy experiment tracking.

## Highlights
- Simple and complex Dense networks vs a 1D CNN
- Linear and non-linear synthetic datasets
- 2-feature and 3-feature variants (adds `activity_level`)
- MLflow tracking: params, metrics per-epoch, artifacts (plots), and saved models
- Clear plots for training curves, predictions, and model complexity

## Project Structure
- `weight-prediction.py`: Main script. Generates data, trains models, logs to MLflow, and saves plots
- `mlruns/`: Local MLflow tracking directory (auto-created)


## Quick Start
From the `weight_prediction` directory:

```bash
python weight-prediction.py
```

## Using MLflow UI
Launch MLflow to browse runs, metrics, and artifacts:

```bash
mlflow ui
```

Then open `http://127.0.0.1:5000` in your browser.

## Key Script Sections (inside `weight-prediction.py`)
- Data generation:
  - `create_synthetic_data()` – 2 features (height, age) with a linear relationship
  - `create_non_linear_data()` – adds non-linear and interaction terms
  - `create_synthetic_data_3features()` / `create_non_linear_data_3features()` – adds `activity_level`
- Preprocessing:
  - `prepare_data()` / `prepare_data_3features()` – train/test split and `StandardScaler`
- Models:
  - `build_dense_model()` – small Dense net
  - `build_complex_dense_model()` – deeper net with Dropout
  - `build_cnn_model()` – 1D CNN consuming features as a short sequence
  - 3-feature counterparts: `build_*_3features()`
- Training & logging:
  - `train_and_evaluate_model()` – trains, evaluates, and logs to MLflow (with `MLflowCallback` for per-epoch metrics)
- Visualization:
  - `plot_training_curves()` / `plot_predictions()`
  - 3-feature versions and a comprehensive comparison plot
- Demonstrations:
  - `test_linear_vs_nonlinear()` – shows when complex models help
  - `compare_model_architectures()` – quick head-to-head
  - `test_3features_comparison()` – linear vs non-linear with 3 features

## Customization
- Sample size: change `n_samples` in `create_*` functions (default 500)
- Epochs: change the `epochs` parameter where models are trained
- Model sizes: edit layers in `build_*` functions
- Validation split: adjustable in the `fit(...)` calls


Saved Keras models are also logged to MLflow under the run’s artifacts.

