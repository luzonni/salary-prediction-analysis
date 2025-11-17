import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("data/salary.csv")

# Criar target numérico
df_reg = df.copy()
df_reg["salary_num"] = df_reg["salary"].apply(lambda x: 1 if ">50K" in x else 0)

# One-hot encoding igual antes
df_reg_encoded = pd.get_dummies(df_reg.drop(columns=["salary"]), drop_first=True)

X = df_reg_encoded.drop("salary_num", axis=1)
y = df_reg_encoded["salary_num"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Regressão Linear
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred = model_lr.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

