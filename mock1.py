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

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

data = pd.read_csv("Galton_Height_Dataset.csv")
print(data.head())
print(data.isna().sum())  # 每一column有多少 NaN
print(data[data.isna().any(axis=1)])  # 列出任一column有 NaN 的 row
# print("Data info: \n", data.info())
# print("\nData description: ", data.describe()) #可以看到 min, max, mean，如果最大值特別大，可能是 outlier。
# print("\nData columns: ", data.columns)

# # Check for missing values
# print("\nMissing values summary:")
# print(data.isnull().sum())

# # Check for duplicate rows
# print("\nDuplicate rows:")
# print(data.duplicated().sum())

# First drop the "rows" for which target contains NaN
data = data.dropna(subset=['Child Height'])
#df.drop(columns=['A'], inplace=True) # example: drop column A
#df = df[~df.isin([np.inf, -np.inf]).any(axis=1)] # drop rows with inf
# df = df.replace([np.inf, -np.inf], np.nan).dropna() # drop rows , 先把 inf 也當作 NaN，再全部丟掉含 NaN 的 row
# Fill Nan
# df = df.fillna(0)  # 全部填0
# df = df.fillna(df.mean())      # 全部欄位各自補平均
# df = df.fillna(df.median())    # 全部欄位各自補中位數
# df['col1'] = df['col1'].fillna(df['col1'].mean()) #指定某個欄位用某個值
# df['col2'] = df['col2'].fillna(123)
# data['Father Height'] = data['Father Height'].fillna(data['Father Height'].mean())
# data['Mother Height'] = data['Mother Height'].fillna(data['Mother Height'].mean())
# data['Number of Siblings'] = data['Number of Siblings'].fillna(data['Number of Siblings'].mean())
#  填補 Inf 為 NaN 再補值
# df = df.replace([np.inf, -np.inf], np.nan)
# df = df.fillna(method='ffill')  # 用前一筆數據填補
# df = df.fillna(method='bfill')  # 用後一筆數據填補
#df = df[~df.isin([np.inf, -np.inf]).any(axis=1)] # if inf, -inf in any column, drop the row
# 把超出範圍的值設為邊界值 (保留範圍內的值)
# df = df[(df['colA'] > -1000) & (df['colA'] < 1000)]
# df['colA'] = df['colA'].clip(-1000, 1000)

# example
# # 1. 讀取資料
# df = pd.read_csv('file.csv')

# # 2. 先把 inf 補成 NaN
# df = df.replace([np.inf, -np.inf], np.nan)

# # 3. 檢查缺失狀況
# print(df.isnull().sum())

# # 4. 決定要填還是刪
# df = df.dropna()             # 全部刪光有缺的
# # 或
# df = df.fillna(df.mean())    # 全部用平均數補

# # 5. 針對部分欄位指定補法
# df['colA'] = df['colA'].fillna(df['colA'].mean())
# df['colB'] = df['colB'].fillna(0)

# Male = 1, Female = 0
data.replace({'M': 1, 'F': 0}, inplace=True)
print(data.head())
# print(data.tail())
# print("'136A' in features:", '136A' in data['Family ID'].values)
# print(features.dtypes)
# features = features.astype(float)
# target = target.astype(float)
# features = np.array(features).astype('float32')
# target = np.array(target).astype('float32')

max_index = pd.to_numeric(data['Family ID'], errors='coerce').max()
data['Family ID'] = data['Family ID'].replace('136A', max_index + 1)
print(data.tail())

# Features and target
features = data[['Family ID', 'Father Height', 'Mother Height', 'Sex', 'Number of Siblings']]
target = data['Child Height']

# 把 Family ID 轉換為數字
features['Family ID'] = pd.to_numeric(features['Family ID'], errors='coerce')

# print(features.dtypes)
# print(type(features['Family ID']))
# 檢查是否有 NaN 或無限值
print("\nNaN in features:", features.isnull().sum().sum())
print("NaN in target:", target.isnull().sum())
print("Inf in features:", np.isinf(features).sum().sum())
print("Inf in target:", np.isinf(target).sum())

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state=42) 
# X_train datatype is dataframe

continuous_X_train = X_train[['Father Height', 'Mother Height']] # continuous_X_train is dataframe
categorical_X_train = X_train[['Sex', 'Number of Siblings']] # categorical_X_train is dataframe

continuous_X_test = X_test[['Father Height', 'Mother Height']]
categorical_X_test = X_test[['Sex', 'Number of Siblings']]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(continuous_X_train) # X_train_scaled is np.ndarray  shape 是 (樣本數, 特徵數)，例如 (600, 2)
X_test_scaled = scaler.transform(continuous_X_test)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_train = encoder.fit_transform(X_train[['Number of Siblings']]) # onehot_train is np.ndarray  shape 是 (樣本數, 特徵數)，例如 (600, 2)
onehot_test = encoder.transform(X_test[['Number of Siblings']])

encoder_fam = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
family_train = encoder_fam.fit_transform(X_train[['Family ID']])
family_test = encoder_fam.transform(X_test[['Family ID']])

# print("continuous_X_train: ", type(continuous_X_train) ) #DF
# print("categorical_X_train: ", type(categorical_X_train)) #DF
# print(type(X_train))           # pandas.DataFrame
# print(type(X_train_scaled))    # numpy.ndarray
# print(type(onehot_train))      # numpy.ndarray 

# X_train_scaled = np.concatenate([X_train_scaled, categorical_X_train], axis = 1) # or np.column_stack
# X_test_scaled = np.concatenate([X_test_scaled, categorical_X_test], axis = 1)
# X_train_scaled = np.column_stack([X_train_scaled, categorical_X_train.values]) # numpy.ndarray and DF
# X_test_scaled = np.column_stack([X_test_scaled, categorical_X_test.values])
X_train_scaled = np.column_stack([family_train, X_train_scaled, X_train[['Sex']].values, onehot_train]) # numpy.ndarray and DF
X_test_scaled = np.column_stack([family_test, X_test_scaled, X_test[['Sex']].values, onehot_test])



print("Final features shape:", X_train_scaled.shape)
print("X_train_scaled:\n", X_train_scaled[:5])
assert not np.isnan(X_train_scaled).any() # 沒有任何 NaN
assert np.isfinite(X_train_scaled).all()    # 全部都是有限值（不是 inf）

# Model
# model = tf.keras.Sequential([
#       tf.keras.layers.Dense(16, activation = 'relu', input_shape = (X_train_scaled.shape[1],)),  #(4,)
#       #tf.keras.layers.Dropout(0.2),
#       tf.keras.layers.Dense(8, activation = 'relu'),
#       #tf.keras.layers.Dropout(0.2),
#       tf.keras.layers.Dense(1)
# ])
# L2 regularization 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],),
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)), # L2
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)), # L1
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = 'mae') #
# train the model
# model.fit(X_train_scaled, y_train, epochs = 100, verbose = 1) #, validation_split=0.1

# early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train, 
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop]
)

loss, mae = model.evaluate(X_test_scaled, y_test)
print(f"Loss: {loss}, MAE: {mae}")

# make predictions
predicted_height = model.predict(X_test_scaled)
predicted_height = predicted_height.flatten() ### Don't forget to flatten the array!!

# calculate mse
MSE = np.mean((predicted_height - y_test) ** 2)
print("MSE: ", MSE)

# calculate r2 score
r2 = r2_score(y_test, predicted_height)
print("R2 score: ", r2)

# plot
plt.figure(figsize = (8, 6))
plt.scatter(y_test, predicted_height)
plt.xlabel("True Height")
plt.ylabel("Predicted Height")

min_val, max_val = min(y_test), max(y_test)
plt.plot([min_val, max_val], [min_val, max_val], color = 'red', label = 'Ideal Line')
plt.legend()
plt.show()

# residual plot
plt.figure()
plt.scatter(y_test, predicted_height - y_test)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Height")
plt.ylabel("Residual (Pred - True)")
plt.title("Residual Plot")
plt.show()
