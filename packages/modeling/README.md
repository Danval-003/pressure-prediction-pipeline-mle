# CRISP-DM – Modeling

Abstracciones para construir, entrenar y evaluar el modelo
LSTM usado en la etapa de **Modeling**.

- `build_lstm_model` crea la arquitectura base.
- `train_lstm_model` incluye *callbacks* (early stopping y checkpoints).
- `evaluate_lstm_model` resume el desempeño del modelo entrenado.

Instala localmente con:

```bash
pip install -e packages/modeling
```
