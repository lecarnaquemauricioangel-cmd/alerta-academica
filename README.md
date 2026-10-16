# Alerta Académica

Sistema de análisis y predicción del rendimiento académico. A partir de datos de estudiantes (horas de estudio, asistencia y tareas entregadas), el proyecto predice la nota final, identifica a quienes están en riesgo de desaprobar y sincroniza los resultados con **Firebase Cloud Firestore**.

---

## Tabla de contenidos

1. [Características](#características)
2. [Arquitectura](#arquitectura)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Requisitos previos](#requisitos-previos)
5. [Instalación](#instalación)
6. [Configuración de credenciales de Firebase](#configuración-de-credenciales-de-firebase)
7. [Ejecución](#ejecución)
8. [Resultados generados](#resultados-generados)
9. [Seguridad: Git y GitHub](#seguridad-git-y-github)
10. [Solución de problemas](#solución-de-problemas)

---

## Características

- **Procesamiento de datos con Pandas**: genera un conjunto de datos sintético reproducible (con semilla) y valida sus columnas antes de usarlo.
- **Regresión lineal con Scikit-learn**: entrena el modelo, lo evalúa con **MAE**, **RMSE** y **R²** y marca con la alerta `revisar` a los estudiantes cuya nota predicha está por debajo de 60.
- **Red neuronal con PyTorch**: entrena un modelo `nn.Linear` durante **1000 épocas** con el optimizador Adam y la pérdida MSE. Normaliza los datos (z-score) y guarda los pesos junto con los parámetros de normalización.
- **Persistencia en la nube con Firestore**: sube estudiantes y predicciones mediante escrituras por lotes idempotentes y guarda un respaldo local de la colección.

## Arquitectura

```
                 ┌──────────────────────┐
                 │       main.py        │   Orquestador (argparse)
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐
│  src/datos.py │  │  src/modelo.py  │  │ src/firebase_repo  │
│  Pandas       │─▶│  Scikit-learn   │─▶│ Cloud Firestore    │
│  CSV + valid. │  │  PyTorch        │  │ (opcional)         │
└───────┬───────┘  └────────┬────────┘  └─────────┬──────────┘
        ▼                   ▼                     ▼
 data/entrada/      data/salida/            Colecciones:
 estudiantes.csv    predicciones.csv        · estudiantes
                    modelos/*.pt            · predicciones
                                            + respaldo_firestore.csv
```

El flujo se ejecuta en cuatro etapas:

| Etapa | Módulo | Descripción |
|-------|--------|-------------|
| 1. Datos | `src/datos.py` | Crea (si no existe) y valida `data/entrada/estudiantes.csv`. |
| 2. Regresión lineal | `src/modelo.py` | Divide los datos en entrenamiento y prueba (80/20), entrena, calcula las métricas y genera las alertas. |
| 3. PyTorch | `src/modelo.py` | Entrena la red durante 1000 épocas y guarda `modelos/modelo_pytorch.pt`. |
| 4. Firestore | `src/firebase_repo.py` | Sube los datos y las predicciones y descarga un respaldo. Solo se ejecuta con `--firebase`. |

**Variables del modelo**

- Características: `horas_estudio`, `asistencia_pct`, `tareas_entregadas`
- Objetivo: `nota_final` (0–100)
- Nota aprobatoria / umbral de alerta: **60**

## Estructura del proyecto

```
alerta_academica/
├── main.py                  # Punto de entrada del proyecto
├── environment.yml          # Definición del entorno Conda
├── analisis.ipynb           # Notebook de análisis exploratorio
├── .gitignore
├── credenciales/            # Llave de Firebase (ignorada por Git)
│   └── firebase_key.json
├── data/
│   ├── entrada/
│   │   └── estudiantes.csv
│   └── salida/              # Archivos generados (ignorados por Git)
├── modelos/                 # Pesos .pt generados (ignorados por Git)
└── src/
    ├── __init__.py
    ├── datos.py             # Generación y validación de datos
    ├── modelo.py            # Scikit-learn y PyTorch
    └── firebase_repo.py     # Conexión y operaciones con Firestore
```

## Requisitos previos

- [Anaconda](https://www.anaconda.com/) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Git
- Un proyecto en [Firebase](https://console.firebase.google.com/) con **Cloud Firestore** habilitado (solo si se va a usar la sincronización)

### Dependencias

Definidas en `environment.yml`:

| Paquete | Uso |
|---------|-----|
| `python=3.12` | Intérprete |
| `numpy`, `pandas` | Manipulación de datos |
| `scikit-learn>=1.4` | Regresión lineal y métricas (`root_mean_squared_error`) |
| `matplotlib` | Gráficos del notebook |
| `jupyterlab` | Entorno del notebook |
| `torch` (pip) | Red neuronal |
| `firebase-admin` (pip) | SDK de administración de Firebase |

## Instalación

1. **Clonar el repositorio**

   ```bash
   git clone <url-del-repositorio>
   cd alerta_academica
   ```

2. **Crear el entorno virtual con Conda**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activar el entorno**

   ```bash
   conda activate alerta_academica
   ```

> En Windows, ejecuta estos comandos desde **Anaconda Prompt**, o inicializa Conda en tu terminal con `conda init`.

## Configuración de credenciales de Firebase

La sincronización con Firestore requiere una **llave de cuenta de servicio**.

1. Entra en la [consola de Firebase](https://console.firebase.google.com/) y selecciona tu proyecto.
2. Ve a **Configuración del proyecto** → **Cuentas de servicio**.
3. Haz clic en **Generar nueva clave privada** y descarga el archivo JSON.
4. Crea la carpeta `credenciales/` dentro de `alerta_academica/` (si no existe):

   ```bash
   mkdir credenciales
   ```

5. Mueve el archivo descargado a esa carpeta y renómbralo como `firebase_key.json`:

   ```
   alerta_academica/credenciales/firebase_key.json
   ```

6. En la consola de Firebase, habilita **Cloud Firestore** (menú **Compilación** → **Firestore Database**).

### Ruta alternativa (opcional)

Si prefieres guardar la llave fuera del proyecto, indica su ubicación con la variable de entorno `FIREBASE_CREDENTIALS`:

```bash
# Linux / macOS
export FIREBASE_CREDENTIALS="/ruta/segura/firebase_key.json"

# Windows (CMD / Anaconda Prompt)
set FIREBASE_CREDENTIALS=C:\ruta\segura\firebase_key.json

# Windows (PowerShell)
$env:FIREBASE_CREDENTIALS = "C:\ruta\segura\firebase_key.json"
```

## Ejecución

Con el entorno activado y desde la carpeta `alerta_academica/`:

### Flujo completo con sincronización en Firestore

```bash
python main.py --firebase
```

### Otras opciones

| Comando | Descripción |
|---------|-------------|
| `python main.py` | Ejecuta el flujo local, sin conectarse a Firebase. |
| `python main.py --firebase` | Ejecuta el flujo completo y sincroniza con Cloud Firestore. |
| `python main.py --regenerar` | Vuelve a generar el CSV sintético. |
| `python main.py --regenerar --n 100 --semilla 7` | Genera 100 estudiantes con la semilla 7. |
| `python main.py --help` | Muestra la ayuda. |

### Salida esperada (resumen)

```
=== 1. Datos ===
[datos] CSV validado: 40 registros, 6 columnas.

=== 2. Regresión lineal (Scikit-Learn) ===
[modelo] Métricas en prueba:
    MAE  : 4.052
    RMSE : 4.969
    R2   : 0.480

=== 3. Demostración PyTorch ===
[pytorch] Entrenando 1000 épocas (lr=0.03)...
    Época 1000 | pérdida: 0.070113

=== 4. Sincronización con Cloud Firestore ===
[firebase] Aplicación inicializada.
[firebase] 40 documentos subidos a 'estudiantes'.
[firebase] 8 documentos subidos a 'predicciones'.

Flujo completado correctamente.
```

## Resultados generados

| Archivo / Colección | Contenido |
|---------------------|-----------|
| `data/salida/predicciones.csv` | Nota real, nota predicha, error y alerta de cada estudiante del conjunto de prueba. |
| `modelos/modelo_pytorch.pt` | Pesos del modelo y parámetros de normalización. |
| `data/salida/respaldo_firestore.csv` | Copia local de la colección `estudiantes` descargada de Firestore. |
| Firestore → `estudiantes` | Un documento por estudiante, con `id_estudiante` como ID. |
| Firestore → `predicciones` | Resultados del modelo sobre el conjunto de prueba. |

## Seguridad: Git y GitHub

La llave de la cuenta de servicio otorga **acceso administrativo completo** a tu proyecto de Firebase. Si se publica, cualquier persona podría leer, modificar o borrar tus datos.

### Protección mediante `.gitignore`

El repositorio incluye un `.gitignore` que excluye las credenciales y los archivos generados:

```gitignore
# Credenciales de Firebase (NUNCA subir)
credenciales/
*firebase*key*.json
*-firebase-adminsdk-*.json
.env

# Artefactos generados
data/salida/*
!data/salida/.gitkeep
modelos/*.pt
```

Así, la carpeta `credenciales/` y cualquier archivo JSON con nombre de llave de Firebase quedan fuera del control de versiones, aunque se encuentren en otra ubicación del proyecto.

### Pasos recomendados antes de subir el proyecto

1. **Inicializa el repositorio** (si aún no existe):

   ```bash
   git init
   ```

2. **Verifica que la llave está ignorada**:

   ```bash
   git check-ignore -v credenciales/firebase_key.json
   ```

   El comando debe mostrar la regla del `.gitignore` que la excluye.

3. **Revisa qué archivos se van a subir**:

   ```bash
   git add .
   git status
   ```

   Confirma que **no** aparece `credenciales/` ni ningún archivo `*.json` con la llave.

4. **Haz el commit y sube el proyecto**:

   ```bash
   git commit -m "Versión inicial de Alerta Académica"
   git branch -M main
   git remote add origin https://github.com/<usuario>/<repositorio>.git
   git push -u origin main
   ```

### Si la llave se subió por error

Eliminar el archivo en un commit posterior **no basta**, porque la llave seguirá en el historial.

1. **Revoca la llave de inmediato** en la consola de Firebase o en Google Cloud (**IAM y administración** → **Cuentas de servicio** → **Claves**) y genera una nueva.
2. Saca el archivo del índice de Git:

   ```bash
   git rm --cached credenciales/firebase_key.json
   git commit -m "Eliminar credenciales del repositorio"
   ```

3. Limpia el historial con herramientas como [`git filter-repo`](https://github.com/newren/git-filter-repo) o [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) si el repositorio ya era público.

### Buenas prácticas adicionales

- No compartas la llave por correo, chat ni capturas de pantalla.
- Usa una llave distinta para cada entorno (desarrollo y producción).
- Revisa periódicamente las claves activas de la cuenta de servicio y elimina las que no uses.
- Activa el **escaneo de secretos** (*secret scanning*) de GitHub en tu repositorio.

## Solución de problemas

| Problema | Solución |
|----------|----------|
| `No se encontró la llave de servicio de Firebase` | Comprueba que existe `credenciales/firebase_key.json` o define `FIREBASE_CREDENTIALS`. |
| `python` abre la Microsoft Store (Windows) | Activa el entorno con `conda activate alerta_academica` antes de ejecutar. |
| El programa se detiene sin mensaje al cargar PyTorch | Ejecuta desde un entorno Conda **activado**, o usa `conda run -n alerta_academica python main.py --firebase`. |
| `ImportError: root_mean_squared_error` | Actualiza Scikit-learn a la versión 1.4 o posterior. |
| Error de permisos en Firestore | Verifica que Cloud Firestore está habilitado y que la llave pertenece al proyecto correcto. |
