from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import RobustScaler

TIMESTEPS_PER_BREATH = 80
DEFAULT_TRAIN_PATH = Path("data/raw")
TRAIN_PART_PATTERN = "train_part_*.csv"

# Número de fragmentos train_part_XXX.csv en GitHub
REMOTE_TRAIN_PART_COUNT = 21

# URL base para descargar datos desde GitHub (raw)
# Ejemplo:
#   https://raw.githubusercontent.com/USER/REPO/BRANCH/data/raw
GITHUB_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Danval-003/pressure-prediction-pipeline-mle"
    "/refs/heads/main/data/raw"
).rstrip("/")


# ============================================================
#   Utilidad: descarga de archivos remotos (GitHub u otros)
# ============================================================

def download_file(url: str, dest: Path, chunk_size: int = 1_048_576) -> Path:
    """
    Descarga un archivo desde `url` a `dest` en chunks para no usar tanta memoria.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Descargando {url} -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return dest


def _remote_file_url(filename: str) -> str:
    """
    Construye la URL remota para un archivo dado su nombre.
    """
    if not GITHUB_DATA_BASE_URL:
        raise FileNotFoundError(
            f"No se encontró el archivo '{filename}' localmente y "
            "GITHUB_DATA_BASE_URL no está configurada."
        )
    return f"{GITHUB_DATA_BASE_URL}/{filename}"


def _ensure_local_file(path: Path) -> Path:
    """
    Garantiza que el archivo exista localmente.
    - Si existe, lo devuelve.
    - Si no existe y hay URL remota, lo descarga.
    """
    if path.exists():
        return path

    url = _remote_file_url(path.name)
    try:
        return download_file(url, path)
    except requests.HTTPError as err:
        # Lo convertimos a FileNotFoundError para poder manejarlo arriba
        raise FileNotFoundError(f"No se pudo descargar {url}: {err}") from err


def _download_remote_train_parts(directory: Path) -> List[Path]:
    """
    Descarga secuencialmente train_part_001.csv ... train_part_NNN.csv
    al directorio `directory` y devuelve la lista de Paths locales.
    """
    if not GITHUB_DATA_BASE_URL:
        return []

    parts: List[Path] = []
    for idx in range(1, REMOTE_TRAIN_PART_COUNT + 1):
        filename = f"train_part_{idx:03d}.csv"
        local_path = directory / filename
        print(f"[load] Asegurando fragmento remoto: {filename}")
        path = _ensure_local_file(local_path)
        parts.append(path)

    return parts


# ============================================================
#   Utilidad: reducir uso de memoria (como en el cuadernito)
# ============================================================

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Misma función del cuadernito, pero SIN usar float16
    (para evitar el error de 'float16 indexes are not supported').
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                # 👇 SOLO float32 / float64, NADA de float16
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    if start_mem > 0:
        print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    return df


# ============================================================
#   Ingeniería de características (como en add_features)
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Misma lógica que add_features(df) en el notebook.
    """
    df = df.copy()

    df["R"] = df["R"].astype(int)
    df["C"] = df["C"].astype(int)

    # Cumulative sum of u_in for each breath
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    # area y area_cumsum
    df["area"] = df["time_step"] * df["u_in"]
    df["area_cumsum"] = df.groupby("breath_id")["area"].cumsum()

    # Lag features para u_in, R, C
    for col in ["u_in", "R", "C"]:
        for lag in range(1, 4):
            df[f"{col}_lag{lag}"] = df.groupby("breath_id")[col].shift(lag)

    # Diff features para time_step, u_in, R, C
    for col in ["time_step", "u_in", "R", "C"]:
        for i in range(1, 5):
            df[f"{col}_diff{i}"] = df.groupby("breath_id")[col].diff(i)

    # Rolling mean y std (ventana=3) para u_in, R, C
    for col in ["u_in", "R", "C"]:
        rolling = df.groupby("breath_id")[col].rolling(window=3, min_periods=1)
        df[f"{col}_rolling_mean"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"{col}_rolling_std"] = rolling.std().reset_index(level=0, drop=True)

    return df


# ============================================================
#   Carga cruda de train.csv (local + opción GitHub multi-part)
# ============================================================

def _resolve_train_files(csv_path: Path) -> List[Path]:
    """
    Determina qué archivos deben cargarse. Acepta:
      - Ruta a archivo único
      - Directorio que contenga fragmentos ``train_part_*.csv``
      - Directorio que contenga un ``train.csv``

    Si no hay nada local pero hay GITHUB_DATA_BASE_URL:
      - Descarga secuencialmente train_part_001.csv ... train_part_NNN.csv
        al directorio indicado.
      - Si no hay fragments remotos (y existiera un train.csv remoto), usaría train.csv.
    """
    # Caso 1: ruta es un archivo concreto (data/raw/train_part_005.csv, etc.)
    if csv_path.is_file():
        return [_ensure_local_file(csv_path)]

    # Vamos a tratar csv_path como directorio lógico (aunque aún no exista)
    # - Si es directorio existente, buscamos cosas locales
    # - Si no existe pero hay URL remota, lo usamos como destino de descargas
    directory = csv_path

    # Si tiene sufijo pero NO es archivo (ej. 'data/raw/train.csv' inexistente),
    # usamos el padre como directorio.
    if csv_path.suffix and not csv_path.is_dir():
        directory = csv_path.parent

    # 1) Preferir datos locales si el directorio existe
    if directory.exists():
        part_files = sorted(directory.glob(TRAIN_PART_PATTERN))
        if part_files:
            print(f"[load] Detectados {len(part_files)} fragmentos locales en {directory}")
            return part_files

        single = directory / "train.csv"
        if single.exists():
            print(f"[load] Detectado train.csv local en {single}")
            return [single]

    # 2) Sin datos locales, intentar remotos si hay URL base
    if GITHUB_DATA_BASE_URL:
        print(f"[load] No hay datos locales en {directory}. Intentando descargar fragmentos remotos...")
        remote_parts = _download_remote_train_parts(directory)
        if remote_parts:
            print(f"[load] Descargados {len(remote_parts)} fragmentos remotos.")
            return remote_parts

        # Como último recurso, intentar un train.csv remoto (si existiera)
        single = directory / "train.csv"
        single = _ensure_local_file(single)
        return [single]

    # 3) Sin datos locales y sin URL remota -> error
    raise FileNotFoundError(
        "No se encontraron archivos de entrenamiento en "
        f"{csv_path}. Asegúrate de tener 'train.csv' o archivos "
        f"{TRAIN_PART_PATTERN} en el directorio, o configurar GITHUB_DATA_BASE_URL."
    )


def load_raw_train_csv(csv_path: Path | str = DEFAULT_TRAIN_PATH) -> pd.DataFrame:
    """
    Carga la data cruda desde la ruta indicada (archivo único o partes).

    Comportamiento:
      - Si los archivos existen localmente, se usan directamente.
      - Si no existen y GITHUB_DATA_BASE_URL está configurada,
        se descargan TODOS los fragmentos train_part_XXX.csv uno por uno
        al directorio especificado y se cargan desde allí.
      - Si no existen y no hay URL remota, lanza FileNotFoundError.
    """
    csv_path = Path(csv_path)
    files = _resolve_train_files(csv_path)

    if len(files) == 1:
        file_path = files[0]
        print(f"[load] Leyendo datos desde {file_path}")
        return pd.read_csv(file_path)

    print(f"[load] Encontrados {len(files)} fragmentos, concatenando...")
    frames = []
    for file_path in files:
        print(f"        - {file_path.name}")
        chunk = pd.read_csv(file_path)
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    return df


# ============================================================
#   API 1: DataFrame preprocesado
# ============================================================

def get_preprocessed_dataframe(
    csv_path: Path | str = DEFAULT_TRAIN_PATH,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con el MISMO preprocesamiento del cuadernito.
    """
    # 1) Carga cruda (local o remota multi-part)
    df = load_raw_train_csv(csv_path=csv_path)

    # 2) Optimización de memoria
    df = reduce_memory_usage(df)

    # 3) Drop 'id'
    df.drop(columns=["id"], inplace=True)

    # 4) Feature engineering
    df = add_features(df)

    # 5) NaN -> 0
    df.fillna(0, inplace=True)

    # 6) Encoding de R y C (mapeos exactos)
    df["R"] = df["R"].map({5: 0, 20: 1, 50: 2})
    df["C"] = df["C"].map({10: 0, 20: 1, 50: 2})

    # 7) Orden consistente
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    return df


# ============================================================
#   API 2: X, y listos para LSTM (3D y 2D)
# ============================================================

def get_lstm_ready_xy(
    csv_path: Path | str = DEFAULT_TRAIN_PATH,
) -> Tuple[np.ndarray, np.ndarray, RobustScaler, List[str]]:
    """
    Devuelve X, y, scaler, features exactamente como en el notebook para el LSTM.
    """
    df = get_preprocessed_dataframe(csv_path=csv_path)

    # === y (target) ===
    y = df[["pressure"]].to_numpy().reshape(-1, TIMESTEPS_PER_BREATH)

    # === X (features) ===
    feature_df = df.drop(columns=["pressure", "breath_id"]).copy()
    features = feature_df.columns.tolist()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(feature_df)  # shape: (num_rows, num_features)

    num_rows, num_features = X_scaled.shape
    if num_rows % TIMESTEPS_PER_BREATH != 0:
        raise ValueError(
            f"El número de filas ({num_rows}) no es múltiplo de {TIMESTEPS_PER_BREATH}."
        )

    X = X_scaled.reshape(-1, TIMESTEPS_PER_BREATH, num_features)

    return X, y, scaler, features


# ============================================================
#   Uso como script: solo mostrar df.head()
# ============================================================

if __name__ == "__main__":
    df_proc = get_preprocessed_dataframe()
    print("\n[main] Shape del DataFrame preprocesado:", df_proc.shape)
    print("\n[main] Primeras filas:")
    print(df_proc.head())
