# data_understanding.py

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo global similar al cuadernito
sns.set(style="whitegrid")
palette = sns.color_palette("viridis", 12)


def dataset_overview(df: pd.DataFrame) -> None:
    """
    Imprime un overview básico del dataset:
    - número de registros y columnas
    - tipos de datos
    - info()
    - describe() de columnas numéricas
    """
    print(f"The given dataset has {df.shape[0]:,} records and {df.shape[1]:,} columns.\n")
    print("dtypes:\n")
    print(df.dtypes)
    print("\nDataFrame.info():")
    df.info()
    print("\nDescriptive statistics (numeric columns):")
    print(df.describe())


def missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """
    Retorna y muestra el conteo de valores faltantes por columna,
    como en la sección de 'Handling Missing Values'.
    """
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing)
    return missing


def duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene las filas duplicadas (si las hay), como en 'Handling Duplicated Values'.
    """
    duplicates = df[df.duplicated(keep=False)]
    print(f"\nNumber of duplicated rows: {len(duplicates)}")
    return duplicates


def plot_column(
    df: pd.DataFrame,
    column: str,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Genera el histograma + boxplot 'bonitos' para una columna numérica,
    tal como en el cuadernito.
    """
    sns.set(style="whitegrid", palette=palette)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    fig.suptitle(f"Distribution and Spread of {column}", fontsize=16, fontweight="bold")

    # Colores
    main_color = palette[0]
    accent_color = palette[6]

    # Histograma + KDE
    sns.histplot(
        df[column],
        kde=True,
        color=main_color,
        edgecolor="black",
        ax=axes[0],
        bins=20,
    )

    axes[0].axvline(df[column].mean(), color=accent_color, linestyle="--", label="Mean")
    axes[0].axvline(df[column].median(), color="purple", linestyle=":", label="Median")

    axes[0].legend()
    axes[0].set_title(f"Histogram of {column}", fontsize=14, fontweight="semibold")
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)

    # Boxplot
    sns.boxplot(
        x=df[column],
        color=main_color,
        width=0.6,
        ax=axes[1],
    )

    mean_val = df[column].mean()
    median_val = df[column].median()

    axes[1].axvline(
        mean_val,
        color=accent_color,
        linestyle="--",
        label=f"Mean: {mean_val:.2f}",
    )
    axes[1].axvline(
        median_val,
        color="purple",
        linestyle=":",
        label=f"Median: {median_val:.2f}",
    )

    axes[1].legend()
    axes[1].set_title(f"Boxplot of {column}", fontsize=14, fontweight="semibold")
    axes[1].set_xlabel(column, fontsize=12)

    # Ajuste final
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if show:
        plt.show()

    return fig, axes


def plot_frequency(
    df: pd.DataFrame,
    column: str,
    show: bool = True,
) -> plt.Figure:
    """
    Genera el gráfico de frecuencias (barplot) para una columna categórica
    (R, C, u_out), igual que en el cuadernito.
    """
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.countplot(data=df, x=column, palette=palette, ax=ax)

    ax.set_title(f"Frequency Plot of {column}", fontsize=14, fontweight="bold")
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def tseries_by_breath(
    df: pd.DataFrame,
    variable: str,
    breath_id: Optional[int] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Grafica una serie de tiempo para la variable dada, opcionalmente filtrada
    por un `breath_id`, igual que en el cuadernito.
    """
    if variable not in df.columns:
        print(f"Error: '{variable}' not found in DataFrame")
        return None

    df_plot = df
    if breath_id is not None:
        df_plot = df[df["breath_id"] == breath_id]

    if df_plot.empty:
        if breath_id is not None:
            print(f"No data found for breath_id = {breath_id}")
        else:
            print("No data found for the specified variable")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_plot["time_step"], df_plot[variable], label=variable, color=palette[0])

    ax.set_xlabel("Time Step")
    ax.set_ylabel(variable)

    title = f"Time Series of `{variable}`"
    if breath_id is not None:
        title += f" for breath_id = {breath_id}"

    ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.legend(loc="upper right")
    ax.grid(True)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def breath_id_summary(df: pd.DataFrame) -> pd.Series:
    """
    Calcula cuántos `breath_id` únicos hay y retorna el conteo por id,
    igual que en la sección 'Unique Breath ID Count'.
    """
    unique_breath = df["breath_id"].nunique()
    print(f"Unique breath_id values: {unique_breath:,}")

    counts = df["breath_id"].value_counts()

    print("\nCounts per unique breath_id value:")
    # Para no imprimir los 75k, mostramos las primeras filas
    print(counts.head())

    return counts


def correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Calcula y grafica la matriz de correlación para las columnas numéricas
    (o para un subconjunto específico, si `columns` se pasa).
    """
    if columns is None:
        num_df = df.select_dtypes(include=["number"])
    else:
        cols = [c for c in columns if c in df.columns]
        num_df = df[cols]

    correlation_matrix = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Matrix", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def run_full_data_understanding(
    df: pd.DataFrame,
    drop_id: bool = True,
    show_plots: bool = True,
) -> None:
    """
    Función helper que reproduce toda la etapa de Data Understanding del cuadernito:
    - overview
    - missing values
    - duplicates
    - distribuciones (R, C, time_step, u_in, u_out, pressure)
    - series de tiempo para algunos breath_id
    - resumen de breath_id
    - matriz de correlación
    """
    # Overview
    dataset_overview(df)

    # Missing y duplicados
    missing_values_summary(df)
    duplicate_rows(df)

    # Trabajar sobre una copia para no mutar el df original
    work_df = df.copy()

    # El cuadernito elimina 'id' en la parte de exploración
    if drop_id and "id" in work_df.columns:
        work_df = work_df.drop(columns=["id"])

    # Distribuciones numéricas (hist + boxplot)
    for col in ["R", "C", "time_step", "u_in", "pressure"]:
        if col in work_df.columns:
            plot_column(work_df, col, show=show_plots)

    # Frecuencias de R, C, u_out (tratadas como categóricas)
    for col in ["R", "C", "u_out"]:
        if col in work_df.columns:
            tmp = work_df.copy()
            tmp[col] = tmp[col].astype(str)
            plot_frequency(tmp, col, show=show_plots)

    # Series de tiempo (pressure, u_in, u_out para algunos breath_id)
    if {"time_step", "pressure", "u_in", "u_out", "breath_id"}.issubset(work_df.columns):
        for bid in [1, 75, 100]:
            tseries_by_breath(work_df, "pressure", breath_id=bid, show=show_plots)
        for bid in [100, 75]:
            tseries_by_breath(work_df, "u_in", breath_id=bid, show=show_plots)
        for bid in [1, 75]:
            tseries_by_breath(work_df, "u_out", breath_id=bid, show=show_plots)

    # Resumen de breath_id
    if "breath_id" in work_df.columns:
        breath_id_summary(work_df)

    # Heatmap de correlación
    correlation_heatmap(work_df, show=show_plots)
