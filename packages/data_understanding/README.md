# CRISP-DM – Data Understanding

Funciones de análisis exploratorio (EDA) y una app de Streamlit
para revisar el dataset de entrenamiento.

Incluye:

- `dataset_overview`, `missing_values_summary`, `correlation_heatmap`, etc.
- `app.py` que construye la interfaz interactiva (`pip install .[app]`).

Instalación local:

```bash
pip install -e packages/data_understanding
```

Para correr la app (modo desarrollo):

```bash
pip install -e packages/data_understanding[app]
python -m streamlit run data_understanding_app.py
```
