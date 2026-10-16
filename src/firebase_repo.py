"""Repositorio de Cloud Firestore: subida y descarga de colecciones."""

import json
import os
from pathlib import Path

import firebase_admin
import pandas as pd
from firebase_admin import credentials, firestore

BASE_DIR = Path(__file__).resolve().parent.parent
# Ruta de la llave de servicio. Se puede sobrescribir con la variable de entorno
# FIREBASE_CREDENTIALS. Nunca subas este archivo al repositorio.
RUTA_CREDENCIALES = Path(
    os.getenv("FIREBASE_CREDENTIALS", BASE_DIR / "credenciales" / "firebase_key.json")
)
LIMITE_LOTE = 500  # Firestore admite como máximo 500 operaciones por batch


def conectar(ruta_credenciales: Path = RUTA_CREDENCIALES):
    """Inicializa Firebase una sola vez y devuelve el cliente de Firestore."""
    if not firebase_admin._apps:
        ruta_credenciales = Path(ruta_credenciales)
        if not ruta_credenciales.exists():
            raise FileNotFoundError(
                f"No se encontró la llave de servicio de Firebase: {ruta_credenciales}. "
                "Descárgala desde la consola de Firebase o define FIREBASE_CREDENTIALS."
            )
        cred = credentials.Certificate(str(ruta_credenciales))
        firebase_admin.initialize_app(cred)
        print("[firebase] Aplicación inicializada.")
    return firestore.client()


def subir_dataframe(df: pd.DataFrame, coleccion: str, campo_id: str = "id_estudiante") -> int:
    """Sube un DataFrame a Firestore mediante escrituras por lotes.

    Si existe la columna `campo_id`, se usa como ID del documento (la subida es
    idempotente); si no, Firestore genera un ID automático.

    Returns:
        Cantidad de documentos escritos.
    """
    if df.empty:
        print(f"[firebase] DataFrame vacío; nada que subir a '{coleccion}'.")
        return 0

    db = conectar()
    # to_json + json.loads convierte tipos de NumPy a tipos nativos de Python.
    registros = json.loads(df.to_json(orient="records", force_ascii=False))
    ref = db.collection(coleccion)

    for inicio in range(0, len(registros), LIMITE_LOTE):
        batch = db.batch()
        for registro in registros[inicio : inicio + LIMITE_LOTE]:
            doc_id = registro.get(campo_id)
            doc = ref.document(str(doc_id)) if doc_id is not None else ref.document()
            batch.set(doc, registro)
        batch.commit()

    print(f"[firebase] {len(registros)} documentos subidos a '{coleccion}'.")
    return len(registros)


def descargar_coleccion(coleccion: str, destino_csv: Path) -> pd.DataFrame:
    """Descarga todos los documentos de una colección y los guarda en un CSV."""
    db = conectar()
    registros = [doc.to_dict() for doc in db.collection(coleccion).stream()]
    df = pd.DataFrame(registros)

    if "id_estudiante" in df.columns:
        df = df.sort_values("id_estudiante").reset_index(drop=True)

    destino_csv = Path(destino_csv)
    destino_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino_csv, index=False, encoding="utf-8")
    print(f"[firebase] {len(df)} documentos de '{coleccion}' respaldados en: {destino_csv}")
    return df
