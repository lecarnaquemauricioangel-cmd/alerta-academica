"""Modelos predictivos de la nota final: Scikit-Learn y PyTorch."""

from pathlib import Path

import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_PREDICCIONES = BASE_DIR / "data" / "salida" / "predicciones.csv"
RUTA_MODELO_PT = BASE_DIR / "modelos" / "modelo_pytorch.pt"

CARACTERISTICAS = ["horas_estudio", "asistencia_pct", "tareas_entregadas"]
OBJETIVO = "nota_final"
UMBRAL_ALERTA = 60


def entrenar_modelo(
    df: pd.DataFrame, ruta_salida: Path = RUTA_PREDICCIONES
) -> tuple[LinearRegression, dict, pd.DataFrame]:
    """Entrena una regresión lineal y genera alertas sobre el conjunto de prueba.

    Returns:
        Tupla (modelo entrenado, diccionario de métricas, DataFrame de predicciones).
    """
    X = df[CARACTERISTICAS]
    y = df[OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    print("[modelo] Métricas en prueba:")
    for nombre, valor in metricas.items():
        print(f"    {nombre:<5}: {valor:.3f}")

    predicciones = df.loc[X_test.index, ["id_estudiante", *CARACTERISTICAS]].copy()
    predicciones["nota_real"] = y_test
    predicciones["nota_predicha"] = y_pred.round(2)
    predicciones["error"] = (predicciones["nota_real"] - predicciones["nota_predicha"]).round(2)
    predicciones["alerta"] = predicciones["nota_predicha"].apply(
        lambda p: "revisar" if p < UMBRAL_ALERTA else "sin_alerta"
    )
    predicciones = predicciones.reset_index(drop=True)

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    predicciones.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"[modelo] Predicciones guardadas en: {ruta_salida}")

    return modelo, metricas, predicciones


def entrenar_pytorch(
    df: pd.DataFrame,
    epocas: int = 1000,
    lr: float = 0.03,
    ruta_modelo: Path = RUTA_MODELO_PT,
    cada: int = 100,
) -> nn.Linear:
    """Demostración: regresión lineal con una capa `nn.Linear` en PyTorch.

    Normaliza características y objetivo (z-score), entrena con MSE + Adam y
    guarda los pesos junto con los parámetros de normalización.
    """
    torch.manual_seed(42)

    X = torch.tensor(df[CARACTERISTICAS].values, dtype=torch.float32)
    y = torch.tensor(df[[OBJETIVO]].values, dtype=torch.float32)

    x_media, x_std = X.mean(dim=0), X.std(dim=0)
    y_media, y_std = y.mean(), y.std()
    X_norm = (X - x_media) / x_std
    y_norm = (y - y_media) / y_std

    modelo = nn.Linear(in_features=3, out_features=1)
    criterio = nn.MSELoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    print(f"[pytorch] Entrenando {epocas} épocas (lr={lr})...")
    for epoca in range(1, epocas + 1):
        optimizador.zero_grad()
        perdida = criterio(modelo(X_norm), y_norm)
        perdida.backward()
        optimizador.step()

        if epoca == 1 or epoca % cada == 0:
            print(f"    Época {epoca:>4} | pérdida: {perdida.item():.6f}")

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": modelo.state_dict(),
            "caracteristicas": CARACTERISTICAS,
            "x_media": x_media,
            "x_std": x_std,
            "y_media": y_media,
            "y_std": y_std,
        },
        ruta_modelo,
    )
    print(f"[pytorch] Modelo guardado en: {ruta_modelo}")
    return modelo


def predecir_pytorch(nuevos: pd.DataFrame, ruta_modelo: Path = RUTA_MODELO_PT) -> pd.Series:
    """Carga el modelo PyTorch guardado y predice notas en la escala original."""
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado: {ruta_modelo}")

    checkpoint = torch.load(ruta_modelo, weights_only=True)
    modelo = nn.Linear(in_features=3, out_features=1)
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()

    X = torch.tensor(nuevos[CARACTERISTICAS].values, dtype=torch.float32)
    X_norm = (X - checkpoint["x_media"]) / checkpoint["x_std"]
    with torch.no_grad():
        y = modelo(X_norm) * checkpoint["y_std"] + checkpoint["y_media"]
    return pd.Series(y.squeeze(1).numpy(), index=nuevos.index, name="nota_predicha_pt")
