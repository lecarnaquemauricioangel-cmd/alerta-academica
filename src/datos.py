"""Generación y validación del dataset sintético de estudiantes."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_CSV = BASE_DIR / "data" / "entrada" / "estudiantes.csv"

COLUMNAS_REQUERIDAS = [
    "id_estudiante",
    "horas_estudio",
    "asistencia_pct",
    "tareas_entregadas",
    "nota_final",
    "estado",
]
NOTA_APROBATORIA = 60


def crear_csv(n: int = 40, semilla: int = 42, ruta: Path = RUTA_CSV) -> pd.DataFrame:
    """Genera `n` estudiantes sintéticos reproducibles y los guarda en CSV.

    Args:
        n: Cantidad de estudiantes a generar.
        semilla: Semilla del generador aleatorio (misma semilla = mismos datos).
        ruta: Ruta del CSV de salida.

    Returns:
        DataFrame con los registros generados.
    """
    if n <= 0:
        raise ValueError("n debe ser un entero positivo.")

    rng = np.random.default_rng(semilla)

    horas = rng.uniform(1, 12, n).round(1)
    asistencia = rng.integers(55, 101, n)  # el límite superior es exclusivo
    tareas = rng.integers(3, 11, n)
    ruido = rng.normal(0, 5, n)

    nota = 15 + 3 * horas + 0.35 * asistencia + 2 * tareas + ruido
    nota = np.clip(nota, 0, 100).round(1)

    df = pd.DataFrame(
        {
            "id_estudiante": [f"E{i:03d}" for i in range(1, n + 1)],
            "horas_estudio": horas,
            "asistencia_pct": asistencia,
            "tareas_entregadas": tareas,
            "nota_final": nota,
            "estado": np.where(nota >= NOTA_APROBATORIA, "aprobado", "en_riesgo"),
        }
    )

    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"[datos] CSV generado con {n} registros en: {ruta}")
    return df


def cargar_y_validar_csv(ruta: Path = RUTA_CSV) -> pd.DataFrame:
    """Carga el CSV de estudiantes y valida su estructura y contenido.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas, está vacío o hay notas fuera de [0, 100].
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    try:
        df = pd.read_csv(ruta, encoding="utf-8")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"El archivo {ruta.name} está vacío.") from e

    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    if df.empty:
        raise ValueError(f"El archivo {ruta.name} no contiene registros.")

    if df["nota_final"].isna().any():
        raise ValueError("Hay registros sin nota_final.")

    fuera_rango = df[(df["nota_final"] < 0) | (df["nota_final"] > 100)]
    if not fuera_rango.empty:
        ids = fuera_rango["id_estudiante"].tolist()
        raise ValueError(f"Notas fuera del rango 0-100 en: {ids}")

    print(f"[datos] CSV validado: {len(df)} registros, {df.shape[1]} columnas.")
    return df
