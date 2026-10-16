"""Orquestador del proyecto de alerta académica.

Uso (desde la carpeta alerta_academica/):
    python main.py                 # flujo local
    python main.py --firebase      # además sincroniza con Cloud Firestore
    python main.py --regenerar     # vuelve a generar el CSV sintético
"""

import argparse
import sys
from pathlib import Path

from src.datos import RUTA_CSV, cargar_y_validar_csv, crear_csv
from src.modelo import entrenar_modelo, entrenar_pytorch

BASE_DIR = Path(__file__).resolve().parent
RUTA_RESPALDO = BASE_DIR / "data" / "salida" / "respaldo_firestore.csv"


def ejecutar(
    sincronizar_firebase: bool = False,
    regenerar: bool = False,
    n: int = 40,
    semilla: int = 42,
) -> None:
    """Ejecuta el flujo completo: datos -> modelos -> (opcional) Firestore."""
    print("=== 1. Datos ===")
    if regenerar or not RUTA_CSV.exists():
        crear_csv(n=n, semilla=semilla)
    df = cargar_y_validar_csv()
    print(df["estado"].value_counts().to_string(), "\n")

    print("=== 2. Regresión lineal (Scikit-Learn) ===")
    _, _, predicciones = entrenar_modelo(df)
    en_alerta = predicciones[predicciones["alerta"] == "revisar"]
    print(f"[main] Estudiantes a revisar en prueba: {len(en_alerta)} de {len(predicciones)}")
    if not en_alerta.empty:
        print(en_alerta[["id_estudiante", "nota_real", "nota_predicha"]].to_string(index=False))
    print()

    print("=== 3. Demostración PyTorch ===")
    entrenar_pytorch(df)
    print()

    if sincronizar_firebase:
        print("=== 4. Sincronización con Cloud Firestore ===")
        # Import diferido: el flujo local funciona aunque firebase-admin no esté configurado.
        from src.firebase_repo import descargar_coleccion, subir_dataframe

        subir_dataframe(df, "estudiantes")
        subir_dataframe(predicciones, "predicciones")
        descargar_coleccion("estudiantes", RUTA_RESPALDO)
    else:
        print("=== 4. Firestore omitido (use --firebase para sincronizar) ===")

    print("\nFlujo completado correctamente.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Proyecto de alerta académica")
    parser.add_argument("--firebase", action="store_true", help="Sincronizar con Cloud Firestore")
    parser.add_argument("--regenerar", action="store_true", help="Regenerar el CSV sintético")
    parser.add_argument("--n", type=int, default=40, help="Cantidad de estudiantes")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    try:
        ejecutar(
            sincronizar_firebase=args.firebase,
            regenerar=args.regenerar,
            n=args.n,
            semilla=args.semilla,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # errores de red/credenciales de Firebase, etc.
        print(f"[error inesperado] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
