# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
# flake8: noqa: E501

import pickle
import gzip
import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

def preparar_dataset(dataframe):
    dataframe = dataframe.rename(columns={"default payment next month": "objetivo"})
    dataframe = dataframe.drop(columns="ID")
    dataframe = dataframe.dropna()
    dataframe = dataframe[(dataframe["SEX"] != 0) & (dataframe["EDUCATION"] != 0) & (dataframe["MARRIAGE"] != 0)]
    dataframe.loc[dataframe["EDUCATION"] > 4, "EDUCATION"] = 4
    return dataframe

def armar_modelo():
    codificacion = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(dtype="int", handle_unknown="ignore"), ["SEX", "EDUCATION", "MARRIAGE"])
        ],
        remainder="passthrough"
    )

    modelo_final = Pipeline(
        steps=[
            ("preproceso", codificacion),
            ("modelo", RandomForestClassifier(random_state=42))
        ],
        verbose=False
    )

    return modelo_final

def buscar_hiperparametros(modelo_base, parametros, n_folds=10):
    buscador = GridSearchCV(
        estimator=modelo_base,
        param_grid=parametros,
        scoring="balanced_accuracy",
        cv=n_folds,
        n_jobs=-1,
    )
    return buscador

def exportar_modelo(objeto_entrenado):
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(objeto_entrenado, archivo)

def obtener_metricas(y_real, y_estimado, nombre_conjunto):
    return {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": precision_score(y_real, y_estimado),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_estimado),
        "recall": recall_score(y_real, y_estimado),
        "f1_score": f1_score(y_real, y_estimado)
    }

def obtener_matriz_confusion(y_real, y_estimado, nombre_conjunto):
    tn, fp, fn, tp = confusion_matrix(y_real, y_estimado).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)}
    }

def guardar_metricas(lista_metricas):
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as salida:
        for linea in lista_metricas:
            salida.write(json.dumps(linea) + "\n")

# Lectura de datos
datos_train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
datos_test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

# Limpieza
datos_train = preparar_dataset(datos_train)
datos_test = preparar_dataset(datos_test)

# Separación
X_train = datos_train.drop(columns="objetivo")
y_train = datos_train["objetivo"]
X_test = datos_test.drop(columns="objetivo")
y_test = datos_test["objetivo"]

# Crear pipeline y entrenar
modelo_base = armar_modelo()
modelo_base.fit(X_train, y_train)

# Optimización
parametros_grid = {
    "modelo__n_estimators": [300, 500],
    "modelo__max_depth": [20, 30],
    "modelo__min_samples_split": [2, 5],
    "modelo__min_samples_leaf": [1, 2]
}
modelo_cv = buscar_hiperparametros(modelo_base, parametros_grid)
modelo_cv.fit(X_train, y_train)

# Guardar modelo entrenado
exportar_modelo(modelo_cv)

# Predicciones
pred_train = modelo_cv.predict(X_train)
pred_test = modelo_cv.predict(X_test)

# Métricas
metricas_train = obtener_metricas(y_train, pred_train, "train")
metricas_test = obtener_metricas(y_test, pred_test, "test")

matriz_train = obtener_matriz_confusion(y_train, pred_train, "train")
matriz_test = obtener_matriz_confusion(y_test, pred_test, "test")

# Guardar resultados
resultados_totales = [metricas_train, metricas_test, matriz_train, matriz_test]
guardar_metricas(resultados_totales)


