# ======================== #
#      1. Imports & Seed   #
# ======================== #
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
import random
import os

# --- Set random seed for reproducibility ---
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# ======================== #
#      2. Data Loading     #
# ======================== #
data = pd.read_csv("Galton_Height_Dataset.csv")
print(data.head())
print(data.isna().sum())                    # Show number of NaNs in each column
print(data[data.isna().any(axis=1)])        # Show rows with any NaN
# print("Data info: \n", data.info())
# print("\nData description: ", data.describe()) #可以看到 min, max, mean，如果最大值特別大，可能是 outlier。
# print("\nData columns: ", data.columns)

# # Check for missing values
# print("\nMissing values summary:")
# print(data.isnull().sum())

# # Check for duplicate rows
# print("\nDuplicate rows:")
# print(data.duplicated().sum())


# ============================= #
#   3. Handling Missing Values  #
# ============================= #

# --- Only drop rows where the target (Child Height) is NaN ---
data = data.dropna(subset=['Child Height'])
# df.drop(columns=['A'], inplace=True)                    # example: drop column A
# df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]        # drop rows with inf
# df = df.replace([np.inf, -np.inf], np.nan).dropna()     # replace inf with NaN, then drop all NaN rows


# --- Different ways to fill missing values ---
# df = df.fillna(0)                                       # fill all missing with 0
# df = df.fillna(df.mean())                               # fill each column with its mean
# df = df.fillna(df.median())                             # fill each column with its median
# df['col1'] = df['col1'].fillna(df['col1'].mean())       # fill a specific column with its mean
# df['col2'] = df['col2'].fillna(123)
# data['Father Height'] = data['Father Height'].fillna(data['Father Height'].mean())
# data['Mother Height'] = data['Mother Height'].fillna(data['Mother Height'].mean())
# data['Number of Siblings'] = data['Number of Siblings'].fillna(data['Number of Siblings'].mean())


# --- Replace inf with NaN, then fill ---
# df = df.replace([np.inf, -np.inf], np.nan)
# df = df.fillna(method='ffill')                          # forward fill
# df = df.fillna(method='bfill')                          # backward fill
# df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]        # drop rows with any inf value

# --- Clip values to a boundary (keep only values within the range) ---
# df = df[(df['colA'] > -1000) & (df['colA'] < 1000)]
# df['colA'] = df['colA'].clip(-1000, 1000)

# --- Example missing value handling workflow ---
# 1. Load data
# df = pd.read_csv('file.csv')
# 2. Replace inf with NaN
# df = df.replace([np.inf, -np.inf], np.nan)
# 3. Check for missing values
# print(df.isnull().sum())
# 4. Decide to drop or fill
# df = df.dropna()             # drop all rows with any missing value
# or
# df = df.fillna(df.mean())    # fill with mean
# 5. Fill specific columns
# df['colA'] = df['colA'].fillna(df['colA'].mean())
# df['colB'] = df['colB'].fillna(0)


# ============================= #
#       4. Data Preprocessing   #
# ============================= #

# --- Sex: Male = 1, Female = 0 ---
data.replace({'M': 1, 'F': 0}, inplace=True)
print(data.head())
# print(data.tail())
# print("'136A' in features:", '136A' in data['Family ID'].values)
# print(features.dtypes)
# features = features.astype(float)
# target = target.astype(float)
# features = np.array(features).astype('float32')
# target = np.array(target).astype('float32')

# --- Convert Family ID with letters to a numeric value ---
max_index = pd.to_numeric(data['Family ID'], errors='coerce').max()
data['Family ID'] = data['Family ID'].replace('136A', max_index + 1)
print(data.tail())


# ============================= #
#    5. Feature Engineering     #
# ============================= #

# --- Select features and target ---
features = data[['Family ID', 'Father Height', 'Mother Height', 'Sex', 'Number of Siblings']].copy()
target = data['Child Height']

# --- Convert Family ID to numeric ---
features['Family ID'] = pd.to_numeric(features['Family ID'], errors='coerce')

# print(features.dtypes)
# print(type(features['Family ID']))

# --- Check for NaN / inf ---
print("\nNaN in features:", features.isnull().sum().sum())
print("NaN in target:", target.isnull().sum())
print("Inf in features:", np.isinf(features).sum().sum())
print("Inf in target:", np.isinf(target).sum())


# ======================== #
#      6. Train/Test Split #
# ======================== #

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state=42) 
# X_train datatype is dataframe

continuous_X_train = X_train[['Father Height', 'Mother Height']]      # continuous features
categorical_X_train = X_train[['Sex', 'Number of Siblings']]          # categorical features

continuous_X_test = X_test[['Father Height', 'Mother Height']]
categorical_X_test = X_test[['Sex', 'Number of Siblings']]


# ============================= #
#    7. Feature Scaling & Encoding
# ============================= #

# --- Standardize continuous features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(continuous_X_train)   # numpy.ndarray, shape=(num_samples, num_features), e.g. (600, 2)
X_test_scaled = scaler.transform(continuous_X_test)

# --- One-hot encoding for categorical features ---
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_train = encoder.fit_transform(X_train[['Number of Siblings']]) # onehot_train is np.ndarray
onehot_test = encoder.transform(X_test[['Number of Siblings']])

encoder_fam = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
family_train = encoder_fam.fit_transform(X_train[['Family ID']])
family_test = encoder_fam.transform(X_test[['Family ID']])

# print("continuous_X_train: ", type(continuous_X_train))   # DataFrame
# print("categorical_X_train: ", type(categorical_X_train)) # DataFrame
# print(type(X_train))           # pandas.DataFrame
# print(type(X_train_scaled))    # numpy.ndarray
# print(type(onehot_train))      # numpy.ndarray 

# --- Combine all features into one array (order: family, scaled continuous, Sex, onehot siblings) ---
# X_train_scaled = np.concatenate([X_train_scaled, categorical_X_train], axis = 1)   # or np.column_stack
# X_test_scaled = np.concatenate([X_test_scaled, categorical_X_test], axis = 1)
# X_train_scaled = np.column_stack([X_train_scaled, categorical_X_train.values])     # numpy.ndarray and DF
# X_test_scaled = np.column_stack([X_test_scaled, categorical_X_test.values])
X_train_scaled = np.column_stack([family_train, X_train_scaled, X_train[['Sex']].values, onehot_train]) # numpy.ndarray and DF
X_test_scaled = np.column_stack([family_test, X_test_scaled, X_test[['Sex']].values, onehot_test])

print("Final features shape:", X_train_scaled.shape)   # (718, 206)
#print("X_train_scaled:\n", X_train_scaled[:5])
assert not np.isnan(X_train_scaled).any()              # No NaN values
assert np.isfinite(X_train_scaled).all()               # All values are finite (no inf)


# ============================= #
#        8. Model Building      #
# ============================= #

# model = tf.keras.Sequential([
#       tf.keras.layers.Dense(16, activation = 'relu', input_shape = (X_train_scaled.shape[1],)),  #(4,)
#       #tf.keras.layers.Dropout(0.2),
#       tf.keras.layers.Dense(8, activation = 'relu'),
#       #tf.keras.layers.Dropout(0.2),
#       tf.keras.layers.Dense(1)
# ])


# --- L2/L1 regularization ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),                         # L2
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),  # L1
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mae') #
# train the model
# model.fit(X_train_scaled, y_train, epochs = 100, verbose = 1) #, validation_split=0.1

# --- Early stopping ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train, 
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop]
)


# ============================= #
#        9. Model Evaluation    #
# ============================= #
loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Loss: {loss}, MAE: {mae}")

# --- Make predictions ---
predicted_height = model.predict(X_test_scaled)
predicted_height = predicted_height.flatten() ### Don't forget to flatten the array!!

# --- Calculate metrics ---
MSE = np.mean((predicted_height - y_test) ** 2)
print("MSE: ", MSE)

r2 = r2_score(y_test, predicted_height) # Coefficient of Determination 0~1, 1 is best R^2 = 1 - (（Sum of Squares of Residuals）/(Sum of Squares of Total Variation))
print("R2 score: ", r2)

# # --- Boxplot ---
# For checking outliers
# plt.boxplot(data['Child Height'])
# plt.title('Boxplot of Child Height')
# plt.show()

# --- Z-score ---
z_scores = np.abs((predicted_height - predicted_height.mean()) / predicted_height.std())
print("Z-scores: ", z_scores)


# ============================= #
#      10. Plotting Results     #
# ============================= #

# --- Predicted vs True values ---
plt.figure(figsize = (8, 6))
plt.scatter(y_test, predicted_height)
plt.xlabel("True Height")
plt.ylabel("Predicted Height")

min_val, max_val = min(y_test), max(y_test)
plt.plot([min_val, max_val], [min_val, max_val], color = 'red', label = 'Ideal Line')
plt.legend()
plt.show()

# --- Residual plot ---
plt.figure()
plt.scatter(y_test, predicted_height - y_test)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Height")
plt.ylabel("Residual (Pred - True)")
plt.title("Residual Plot")
plt.show()
