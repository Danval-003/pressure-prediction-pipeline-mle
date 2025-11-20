# Paquetes CRISP-DM

El repositorio está organizado como un monorepo con un paquete Python
por cada etapa relevante del ciclo CRISP-DM.

```
packages/
  data_preparation/
  data_understanding/
  modeling/
  evaluating/
  deployment/
```

## Estructura

- Cada paquete tiene `pyproject.toml`, `README.md` y el código dentro de
  `src/crispdm_<etapa>/`.
- Los módulos superiores (`data_preparation.py`, `modeling.py`, etc.)
  actúan como *wrappers* para mantener compatibilidad con scripts
  existentes; internamente importan `local_dev.py`, que inserta
  `packages/*/src` en `sys.path`.

## Instalación local

```bash
pip install -e packages/data_preparation
pip install -e packages/data_understanding[app]
pip install -e packages/modeling
pip install -e packages/evaluating
pip install -e packages/deployment
```

El sufijo `[app]` instala Streamlit para la app de *Data Understanding*.

## Dependencias cruzadas

- `crispdm_evaluating` importa funciones de `crispdm_data_preparation` y
  `crispdm_modeling`.
- `crispdm_deployment` reutiliza `add_features` / `get_lstm_ready_xy`
  para garantizar el mismo pipeline de datos durante la inferencia.

Todas estas dependencias están declaradas en los respectivos
`pyproject.toml`, por lo que `pip install ./packages/<etapa>` resuelve
las versiones correctas.

## CI/CD

- `.github/workflows/ci.yml` construye cada paquete (`python -m build`)
  en una matriz cuando hay `push` a `main` o `pull_request`.
- `.github/workflows/release.yml` se activa con tags estilo
  `data-preparation-vMAJOR.MINOR.PATCH` y publica los artefactos
  (`.whl`, `.tar.gz`) en un release de GitHub, validando que el tag
  coincida con la versión declarada en `pyproject.toml`.

## Flujo de versionamiento recomendado

1. Ajusta la versión en el `pyproject.toml` del paquete.
2. Haz commit/push de los cambios.
3. Crea el tag `etapa-vX.Y.Z` y súbelo (`git push origin et...`).
4. Verifica que el workflow `release` haya adjuntado los artefactos al
   release correspondiente.
