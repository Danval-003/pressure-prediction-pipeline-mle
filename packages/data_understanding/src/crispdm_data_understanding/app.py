# data_understanding_app.py

from pathlib import Path

import streamlit as st
import pandas as pd

from . import (
    dataset_overview,
    missing_values_summary,
    duplicate_rows,
    plot_column,
    plot_frequency,
    tseries_by_breath,
    breath_id_summary,
    correlation_heatmap,
)

st.set_page_config(
    page_title="Ventilator Pressure - Data Understanding",
    layout="wide",
)

# =========================================================
# Carga de datos
# =========================================================

PART_PATTERN = "train_part_*.csv"


def _load_train_data(path: str | Path) -> pd.DataFrame:
    base = Path(path)

    if base.is_dir():
        parts = sorted(base.glob(PART_PATTERN))
        if parts:
            frames = [pd.read_csv(part) for part in parts]
            return pd.concat(frames, ignore_index=True)
        candidate = base / "train.csv"
        if candidate.exists():
            return pd.read_csv(candidate)
    elif base.is_file():
        return pd.read_csv(base)
    else:
        parts = sorted(base.parent.glob(PART_PATTERN))
        if parts:
            frames = [pd.read_csv(part) for part in parts]
            return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(
        f"No se encontraron archivos en {path}. "
        "Coloca train.csv o fragmentos train_part_*.csv en data/raw/."
    )


@st.cache_data
def load_data(path: str | Path = "data/raw") -> pd.DataFrame:
    return _load_train_data(path)


df = load_data()

# =========================================================
# 1. Overview del dataset
# =========================================================

st.title("Ventilator Pressure Prediction 💨 – Data Understanding")

st.header("1. Dataset Overview and Description")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 1.1 Vista general del dataset")
    st.markdown(f"**Shape:** `{df.shape[0]:,}` filas × `{df.shape[1]}` columnas")
    st.markdown("**Primeras filas**")
    st.dataframe(df.head())

with col2:
    st.markdown("### 1.2 Tipos de datos y estadísticas básicas")
    st.markdown("**dtypes**")
    st.text(df.dtypes.to_string())
    st.markdown("**Estadísticos descriptivos (columnas numéricas)**")
    st.dataframe(df.describe())

st.markdown(
    """
**Observaciones 💡**  

- El archivo `train.csv` contiene alrededor de **6,036,000 registros y 8 columnas**, por lo que se considera un conjunto de datos grande y suficientemente detallado para modelar comportamiento respiratorio en ventilación mecánica.  
- Cada fila corresponde a un **paso de tiempo** dentro de una respiración, y los **ciclos de respiración** se identifican por `breath_id`.  
- Cada respiración dura aproximadamente **3 segundos**, discretizada en múltiples `time_step`, lo que permite tratar el problema como una **serie de tiempo**.  
- Para cada `breath_id` se observan las señales de control (`u_in`, `u_out`) y la respuesta del sistema (`pressure`), además de atributos `R` y `C` que representan propiedades pulmonares.
"""
)

# =========================================================
# 2. Clasificación de variables
# =========================================================

st.header("2. Variable Classification and Description")

var_info = pd.DataFrame(
    {
        "Nombre": [
            "id",
            "breath_id",
            "R",
            "C",
            "time_step",
            "u_in",
            "u_out",
            "pressure",
        ],
        "Descripción": [
            "Identificador único de cada registro (time step global).",
            "Identificador de cada ciclo de respiración.",
            "Resistencia de las vías respiratorias (cmH2O/L/s).",
            "Cumplimiento (compliance) del pulmón (mL/cmH2O).",
            "Marca de tiempo para cada medición dentro de la respiración.",
            "Control de entrada de aire (0–100: apertura de válvula inspiratoria).",
            "Control de salida de aire (0 o 1: válvula espiratoria cerrada/abierta).",
            "Presión en las vías respiratorias (cmH2O).",
        ],
        "Tipo de variable": [
            "Cualitativa (Nominal)",
            "Cualitativa (Nominal)",
            "Cuantitativa (Discreta, pero con comportamiento categórico)",
            "Cuantitativa (Discreta, pero con comportamiento categórico)",
            "Cuantitativa (Continua)",
            "Cuantitativa (Discreta)",
            "Cualitativa (Nominal)",
            "Cuantitativa (Continua)",
        ],
    }
)

st.dataframe(var_info)

st.markdown(
    """
**Observaciones 💡**  

- El dataset combina **3 variables cualitativas nominales** (`id`, `breath_id`, `u_out`) y **5 cuantitativas** (`R`, `C`, `time_step`, `u_in`, `pressure`).  
- Aunque `R` y `C` se almacenan como números, **solo toman unos pocos valores discretos** (5, 20, 50 en R; 10, 20, 50 en C), por lo que su comportamiento es **claramente categórico**.  
- Entender esta clasificación es clave para decidir más adelante qué columnas se **codifican como categorías**, cuáles se escalan y cuáles se descartan (por ejemplo, `id` como identificador puramente técnico).
"""
)

# =========================================================
# 3. Missing values & Duplicates
# =========================================================

st.header("3. Data Cleaning: Missing Values & Duplicates")

st.subheader("3.1 Missing values")

mv = missing_values_summary(df)
st.dataframe(mv.to_frame("missing_count"))

st.markdown(
    """
**Observaciones 💡**  

- Todas las columnas presentan **0 valores faltantes**.  
- Esto significa que **no es necesario aplicar técnicas de imputación** (ni media, mediana ni modelos de imputación), lo cual simplifica la etapa de preprocesamiento.  
"""
)

st.subheader("3.2 Duplicated rows")

dups = duplicate_rows(df)
st.write(f"**Número de filas duplicadas:** `{len(dups)}`")
if len(dups) > 0:
    st.dataframe(dups.head())
else:
    st.info("No se encontraron filas duplicadas en el conjunto de datos.")

st.markdown(
    """
**Observaciones 💡**  

- No se detectan filas duplicadas, por lo que **no es necesario aplicar estrategias de deduplicación**.  
- Esto refuerza que cada combinación (`id`, `time_step`) describe un **estado único en el tiempo** para una respiración.  
"""
)

# =========================================================
# 4. Distribuciones y frecuencias
# =========================================================

st.header("4. Data Visualization and Distribution Analysis")

st.subheader("4.1 Distribuciones numéricas (Histogram + Boxplot)")

numeric_cols_to_plot = ["R", "C", "time_step", "u_in", "pressure"]
for col in numeric_cols_to_plot:
    if col in df.columns:
        st.markdown(f"#### {col}")
        fig, _ = plot_column(df, col, show=False)
        st.pyplot(fig)

        # Comentarios específicos por variable
        if col == "R":
            st.markdown(
                """
**Observaciones 💡 (R)**  

- `R` solo toma valores **5, 20 y 50**, por lo que **no se comporta como variable continua**, sino como un conjunto de niveles de resistencia.  
- Esto motiva tratar `R` como **variable categórica** y no como numérica pura en pasos posteriores (por ejemplo, mapeo a {0, 1, 2}).  
"""
            )
        elif col == "C":
            st.markdown(
                """
**Observaciones 💡 (C)**  

- `C` solo toma valores **10, 20 y 50**, representando distintos niveles de compliance pulmonar.  
- Al igual que `R`, `C` se interpreta mejor como una **categoría** que como un número continuo.  
"""
            )
        elif col == "time_step":
            st.markdown(
                """
**Observaciones 💡 (time_step)**  

- `time_step` se distribuye de forma **uniforme entre 0 y ~3 segundos**, con media y mediana cercanas a 1.31.  
- No se observan valores atípicos, lo cual indica una discretización uniforme y estable del tiempo en cada respiración.  
"""
            )
        elif col == "u_in":
            st.markdown(
                """
**Observaciones 💡 (u_in)**  

- La mayor parte de los valores de `u_in` se concentra entre **0 y 4**, pero existen **picos muy altos** que actúan como outliers.  
- Este comportamiento sugiere que la señal de entrada tiene **pulsos de alta intensidad**, lo que justifica el uso posterior de escaladores robustos como `RobustScaler` para reducir el impacto de valores extremos.  
"""
            )
        elif col == "pressure":
            st.markdown(
                """
**Observaciones 💡 (pressure)**  

- La mayoría de valores de `pressure` se agrupa entre **0 y 10 cmH2O**, con algunos **valores atípicos** más altos.  
- El patrón de la distribución es consistente con una señal fisiológica que varía dentro de un rango controlado, con picos puntuales de presión.  
"""
            )

st.subheader("4.2 Frecuencias de variables categóricas (R, C, u_out)")

# R
if "R" in df.columns:
    st.markdown("##### Frecuencia de R (como categoría)")
    tmp = df.copy()
    tmp["R"] = tmp["R"].astype(str)
    fig_R = plot_frequency(tmp, "R", show=False)
    st.pyplot(fig_R)
    st.markdown(
        """
**Observaciones 💡**  

- Se confirma que `R` está concentrada solo en **tres valores discretos (5, 20, 50)**.  
- Esta concentración refuerza la decisión de tratarla como **variable categórica** y codificarla con un **mapeo simple** (por ejemplo, {5 → 0, 20 → 1, 50 → 2}).  
"""
    )

# C
if "C" in df.columns:
    st.markdown("##### Frecuencia de C (como categoría)")
    tmp = df.copy()
    tmp["C"] = tmp["C"].astype(str)
    fig_C = plot_frequency(tmp, "C", show=False)
    st.pyplot(fig_C)
    st.markdown(
        """
**Observaciones 💡**  

- `C` también se concentra en **tres niveles discretos (10, 20, 50)**.  
- Igual que con `R`, esto justifica mapear `C` a una representación numérica compacta **{10 → 0, 20 → 1, 50 → 2}**, preservando el orden jerárquico de la compliance.  
"""
    )

# u_out
if "u_out" in df.columns:
    st.markdown("##### Frecuencia de u_out")
    tmp = df.copy()
    tmp["u_out"] = tmp["u_out"].astype(str)
    fig_uout = plot_frequency(tmp, "u_out", show=False)
    st.pyplot(fig_uout)
    st.markdown(
        """
**Observaciones 💡**  

- `u_out` está **fuertemente desbalanceada**, predominando el valor `"1"` (salida de aire permitida).  
- Esto indica que el sistema pasa buena parte del tiempo en **fase espiratoria**, lo que influye en la dinámica de `pressure` y en cómo se interpretan las correlaciones con esta variable.  
"""
    )

# =========================================================
# 5. Series de tiempo por breath_id
# =========================================================

st.header("5. Time Series Analysis by breath_id")

if {"time_step", "breath_id"}.issubset(df.columns):
    unique_breath_ids = df["breath_id"].unique()
    unique_breath_ids_sorted = sorted(unique_breath_ids.tolist())

    st.markdown(
        """
Seleccioná un `breath_id` y una variable para visualizar la serie de tiempo
correspondiente a esa respiración.
"""
    )

    col_left, col_right = st.columns(2)
    with col_left:
        selected_breath = st.selectbox(
            "breath_id",
            unique_breath_ids_sorted,
            index=0,
        )

    with col_right:
        variable = st.selectbox(
            "Variable",
            [col for col in ["pressure", "u_in", "u_out"] if col in df.columns],
        )

    fig_ts = tseries_by_breath(df, variable, breath_id=int(selected_breath), show=False)
    if fig_ts is not None:
        st.pyplot(fig_ts)

    st.markdown(
        """
**Observaciones 💡**  

- Al analizar `pressure` a lo largo del tiempo en un `breath_id`, se observa que la presión suele ser **moderadamente alta durante el primer segundo**, y luego **disminuye de forma gradual** en los segundos siguientes.  
- En el caso de `u_in`, el patrón típico es una **entrada de aire relativamente alta al inicio**, que luego decrece hasta casi cero antes de iniciar un nuevo ciclo.  
- Para `u_out`, se aprecia que la válvula de salida suele **activarse después de ~1 segundo**, favoreciendo la fase de espiración y el descenso de la presión.  
- Estos patrones confirman la **naturaleza fuertemente temporal** del problema y justifican el uso de **modelos secuenciales** como LSTM/BiLSTM en etapas posteriores.  
"""
    )
else:
    st.warning("No se encontraron las columnas necesarias para el análisis de series de tiempo.")

# =========================================================
# 6. Resumen de breath_id
# =========================================================

st.header("6. Unique Breath ID Count")

if "breath_id" in df.columns:
    counts = breath_id_summary(df)
    st.markdown("**Número de breath_id únicos**")
    st.write(f"{df['breath_id'].nunique():,}")
    st.markdown("**Conteos (primeros 20)**")
    st.dataframe(counts.head(20))

    st.markdown(
        """
**Observaciones 💡**  

- Existen alrededor de **75,450 respiraciones únicas (`breath_id`)**, cada una con **exactamente 80 pasos de tiempo**.  
- Esta estructura homogénea permite **reorganizar los datos** en tensores de forma `(n_breaths, 80, n_features)` y aplicar técnicas como:  
  - **Lags (retrasos)** por respiración.  
  - **Estadísticas móviles (rolling)**.  
  - **Acumulados** (por ejemplo, `u_in_cumsum`, `area_cumsum`).  
"""
    )
else:
    st.warning("La columna 'breath_id' no está presente en el DataFrame.")

# =========================================================
# 7. Matriz de correlación
# =========================================================

st.header("7. Correlation Matrix (Heatmap)")

fig_corr = correlation_heatmap(df, show=False)
st.pyplot(fig_corr)

st.markdown(
    """
**Observaciones 💡**  

- Las variables `time_step` y `u_out` muestran una **correlación negativa moderada con `pressure`** (coeficientes menores a -0.5 en valor absoluto).  
- Esto sugiere que, a medida que avanza el tiempo o la válvula de salida está abierta, la presión tiende a **disminuir**, lo cual coincide con la interpretación clínica esperada.  
- La relación más fuerte se observa entre `time_step` y `u_out`, con un coeficiente superior a **0.8**, indicando que la activación de la válvula de salida ocurre de forma **altamente sincronizada con el avance del tiempo dentro del ciclo respiratorio**.  
"""
)

# =========================================================
# 8. Implicaciones para la ingeniería de características
# =========================================================

st.header("8. Implicaciones para Feature Engineering ⚙")

st.markdown(
    """
A partir del análisis exploratorio realizado, se justifican las siguientes decisiones de 
ingeniería de características (que se implementan en la etapa de *Data Preparation*):

- **Suma acumulada de `u_in` y `area`** (`u_in_cumsum`, `area_cumsum`): permiten capturar el **efecto acumulado** de la entrada de aire y de la “energía” inyectada al sistema a lo largo del tiempo.  
- **Lags de `u_in`, `R` y `C`** (hasta 3 retrasos): permiten que el modelo considere explícitamente los **valores recientes** de las mismas variables para cada respiración, lo cual es estándar en modelado de series de tiempo.  
- **Diferencias (`diff`) de `time_step`, `u_in`, `R` y `C`**: ayudan a modelar **cambios abruptos y dinámicas no estacionarias**, muy visibles en la serie de `u_in`.  
- **Estadísticas móviles (rolling mean y std)**: capturan la **tendencia local y la variabilidad reciente** de las señales, lo que aporta contexto adicional al modelo.  
- **Codificación de `R` y `C` como {0, 1, 2}**: simplifica su uso en el modelo, respetando la jerarquía física (valores más altos representan mayor resistencia/compliance).  

En conjunto, estas decisiones están directamente motivadas por los patrones observados en el EDA y apuntan a que el modelo LSTM pueda aprender **tanto el nivel** de las señales como sus **cambios en el tiempo**.
"""
)
