import tkinter as tk
from tkinter import ttk
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import numpy as np
import pandas as pd

from part4_regressaologistica import (
    X_train, X_test,
    y_train, y_test,
    feature_names
)

# ===================================================================
# 1) REGRESSÃO LINEAR
# ===================================================================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_reg = lin_reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_reg)
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_reg)

coef_df = pd.DataFrame({
    "Variável": feature_names,
    "Coeficiente": lin_reg.coef_
})

# ===================================================================
# 2) REGRESSÃO LOGÍSTICA
# ===================================================================
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)

y_pred_clf = log_reg.predict(X_test)
y_prob_clf = log_reg.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred_clf)
prec = precision_score(y_test, y_pred_clf)
rec = recall_score(y_test, y_pred_clf)
f1 = f1_score(y_test, y_pred_clf)
auc = roc_auc_score(y_test, y_prob_clf)
cm = confusion_matrix(y_test, y_pred_clf)

odds_ratios = pd.DataFrame({
    "Variável": feature_names,
    "Coeficiente Logístico": log_reg.coef_[0],
    "Odds Ratio": np.exp(log_reg.coef_[0])
})

# ===================================================================
# 3) INTERFACE GRÁFICA TKINTER
# ===================================================================

root = tk.Tk()
root.title("Resultados dos Modelos de Regressão e Classificação")
root.geometry("900x700")

notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# ----------------------------------------------------
# Função para criar tabelas em Treeview
# ----------------------------------------------------
def criar_tabela(frame, df):
    cols = list(df.columns)
    tree = ttk.Treeview(frame, columns=cols, show="headings")
    
    for col in cols:
        tree.heading(col, text=col)
        tree.column(col, width=150, anchor="center")

    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))
    
    tree.pack(fill="both", expand=True)

# ----------------------------------------------------
# Aba 1 — Métricas de Regressão Linear
# ----------------------------------------------------
frame_reg = ttk.Frame(notebook)
notebook.add(frame_reg, text="Regressão Linear")

tk.Label(frame_reg, text=f"MAE: {mae:.4f}").pack()
tk.Label(frame_reg, text=f"MSE: {mse:.4f}").pack()
tk.Label(frame_reg, text=f"RMSE: {rmse:.4f}").pack()
tk.Label(frame_reg, text=f"R²: {r2:.4f}").pack()

# Tabela coeficientes
criar_tabela(frame_reg, coef_df)

# ----------------------------------------------------
# Aba 2 — Métricas de Regressão Logística
# ----------------------------------------------------
frame_log = ttk.Frame(notebook)
notebook.add(frame_log, text="Regressão Logística")

tk.Label(frame_log, text=f"Acurácia: {acc:.4f}").pack()
tk.Label(frame_log, text=f"Precisão: {prec:.4f}").pack()
tk.Label(frame_log, text=f"Recall: {rec:.4f}").pack()
tk.Label(frame_log, text=f"F1-score: {f1:.4f}").pack()
tk.Label(frame_log, text=f"AUC-ROC: {auc:.4f}").pack()

# Matriz de Confusão
cm_df = pd.DataFrame(cm, columns=["Previsto 0", "Previsto 1"], index=["Real 0", "Real 1"])
criar_tabela(frame_log, cm_df)

# Odds Ratios
criar_tabela(frame_log, odds_ratios)

root.mainloop()
