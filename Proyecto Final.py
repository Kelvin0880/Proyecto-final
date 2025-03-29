
"""
Tarea: Predicción de precios de viviendas en India
Modelos: Regresión Lineal vs. Regresión Logística (clasificación binaria)
"""

# =============================================================================
# 1. Importar librerías
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix

# Configurar estilo de gráficos
sns.set(style="whitegrid")

# =============================================================================
# 2. Cargar y explorar datos
# =============================================================================
# Cargar dataset de entrenamiento
train = pd.read_csv("train.csv")  # filepath: c:\Users\kelvi\OneDrive\Escritorio\Proyecto final\train.csv
test = pd.read_csv("test.csv")    # filepath: c:\Users\kelvi\OneDrive\Escritorio\Proyecto final\test.csv

# Mostrar estructura inicial
print("=== Primeras filas del dataset ===")
print(train.head())
print("\n=== Información de columnas ===")
print(train.info())
print("\n=== Estadísticas descriptivas ===")
print(train.describe())

# Ver valores nulos
print("\n=== Valores nulos por columna ===")
print(train.isnull().sum())

# Rellenar valores nulos 
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# =============================================================================
# 3. Análisis Exploratorio (EDA)
# =============================================================================
# Distribución del precio (variable objetivo)
plt.figure(figsize=(10, 6))
sns.histplot(train["TARGET(PRICE_IN_LACS)"], kde=True, bins=50, color="blue")
plt.title("Distribución de TARGET(PRICE_IN_LACS) (Precio en Lacs)")
plt.xlabel("Precio (en Lacs)")
plt.ylabel("Frecuencia")
plt.show()

# Correlación entre variables numéricas
numeric_cols = train.select_dtypes(include=np.number).columns
plt.figure(figsize=(10, 6))
sns.heatmap(train[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación (Variables Numéricas)")
plt.show()

# =============================================================================
# 4. Preprocesamiento de datos
# =============================================================================
# Definir variables
numeric_features = ["SQUARE_FT", "LONGITUDE", "LATITUDE", "BHK_NO."]
categorical_features = ["POSTED_BY", "BHK_OR_RK", "READY_TO_MOVE", "RERA"]

# Pipeline de transformación
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Separar features y target
X = train.drop("TARGET(PRICE_IN_LACS)", axis=1)
y = train["TARGET(PRICE_IN_LACS)"]

# =============================================================================
# 5. Modelado: Regresión Lineal
# =============================================================================
# Dividir datos (para regresión lineal)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar pipeline
model_linear = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
model_linear.fit(X_train, y_train)

# Evaluar modelo
y_pred_linear = model_linear.predict(X_test)
mse = mean_squared_error(y_test, y_pred_linear)
mae = mean_absolute_error(y_test, y_pred_linear)
r2 = r2_score(y_test, y_pred_linear)

print("\n=== Resultados Regresión Lineal ===")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Gráfico de predicciones vs reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.5, color="green")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "--k", linewidth=2)
plt.xlabel("Precio Real (Lacs)")
plt.ylabel("Precio Predicho (Lacs)")
plt.title("Regresión Lineal: Real vs. Predicho")
plt.show()

# =============================================================================
# 6. Modelado: Regresión Logística (Clasificación Binaria)
# =============================================================================
# Convertir y en variable binaria (1 si precio > mediana, 0 si no)
y_binary = (y > y.median()).astype(int)

# Redividir datos (para clasificación)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Crear y entrenar pipeline
model_logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
model_logistic.fit(X_train, y_train)

# Evaluar modelo
y_pred_logistic = model_logistic.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_logistic)
report = classification_report(y_test, y_pred_logistic)

print("\n=== Resultados Regresión Logística (Clasificación) ===")
print(f"Accuracy: {accuracy:.2f}")
print("\nReporte de Clasificación:\n", report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_logistic)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Regresión Logística")
plt.show()

# =============================================================================
# 7. Predicciones en el conjunto de prueba
# =============================================================================
# Usar el modelo de regresión lineal para predecir precios en el conjunto de prueba
test_predictions = model_linear.predict(test)

# Guardar las predicciones en un archivo CSV
submission = pd.DataFrame({"TARGET(PRICE_IN_LACS)": test_predictions})
submission.to_csv("submission.csv", index=False)
print("\nPredicciones guardadas en 'submission.csv'.")

# =============================================================================
# 8. Comparación Final
# =============================================================================
print("\n=== Comparación de Modelos ===")
print(f"""
Regresión Lineal:
  - MSE: {mse:.2f}
  - MAE: {mae:.2f}
  - R²: {r2:.2f}

Regresión Logística (Clasificación):
  - Accuracy: {accuracy:.2f}
  - F1-Score: {classification_report(y_test, y_pred_logistic, output_dict=True)['weighted avg']['f1-score']:.2f}
""")