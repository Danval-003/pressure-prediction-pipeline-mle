# CRISP-DM – Data Preparation

Utilidades reutilizables para la etapa de **Data Preparation** del flujo
CRISP-DM para _Ventilator Pressure Prediction_. El paquete expone
funciones para:

- Optimizar memoria del `DataFrame` original.
- Construir todas las _feature engineering_ requeridas.
- Generar `X`, `y`, escaladores y lista de *features* para modelos LSTM.

Instala el paquete (localmente) con:

```bash
pip install -e packages/data_preparation
```

Luego puedes importar las funciones mediante:

```python
from crispdm_data_preparation import get_preprocessed_dataframe
```
