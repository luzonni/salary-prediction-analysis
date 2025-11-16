from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Criar target numérico igual à Parte A
df_log = df.copy()
df_log["salary_num"] = df_log["salary"].apply(lambda x: 1 if ">50K" in x else 0)

# One-hot encoding
df_log_encoded = pd.get_dummies(df_log.drop(columns=["salary"]), drop_first=True)

# Separação em X e y
X = df_log_encoded.drop("salary_num", axis=1)
y = df_log_encoded["salary_num"]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo de Regressão Logística
model_log = LogisticRegression(max_iter=1000)
model_log.fit(X_train, y_train)

# Previsões
y_pred = model_log.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Acurácia:", acc)
print("\nMatriz de Confusão:\n", cm)
print("\nRelatório de Classificação:\n", report)