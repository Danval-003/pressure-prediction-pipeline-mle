# Personal_asa – CRISP-DM Packages

Este repositorio ahora está organizado como un **monorepo de paquetes**
Python, uno por cada etapa aplicable del ciclo CRISP-DM:

| Etapa | Distribución | Módulo |
| --- | --- | --- |
| Data Preparation | `crispdm-data-preparation` | `crispdm_data_preparation` |
| Data Understanding | `crispdm-data-understanding` | `crispdm_data_understanding` |
| Modeling | `crispdm-modeling` | `crispdm_modeling` |
| Evaluation | `crispdm-evaluating` | `crispdm_evaluating` |
| Deployment | `crispdm-deployment` | `crispdm_deployment` |

Cada paquete vive dentro de `packages/<etapa>` con un `pyproject.toml`
propio y un árbol `src/` listo para publicarse.

## Uso local

Para evitar instalaciones repetidas durante el desarrollo se añadió
`local_dev.py`, que inserta todos los `packages/*/src` en `sys.path`.
Los archivos de compatibilidad (`data_preparation.py`, `modeling.py`,
etc.) importan este módulo automáticamente, por lo que puedes seguir
ejecutándolos como scripts si estás dentro del repositorio.

Si prefieres instalarlos de forma editable:

```bash
pip install -e packages/data_preparation
pip install -e packages/data_understanding[app]
pip install -e packages/modeling
pip install -e packages/evaluating
pip install -e packages/deployment
```

## CI/CD

- `.github/workflows/ci.yml` construye **todos** los paquetes en cada
  `push` a `main` y en cada `pull_request`.
- `.github/workflows/release.yml` publica automáticamente el paquete
  correspondiente cuando se empuja un tag con el formato:

  ```
  data-preparation-vMAJOR.MINOR.PATCH
  data-understanding-vMAJOR.MINOR.PATCH
  modeling-vMAJOR.MINOR.PATCH
  evaluating-vMAJOR.MINOR.PATCH
  deployment-vMAJOR.MINOR.PATCH
  ```

  El workflow valida que la versión declarada en `pyproject.toml`
  coincida con la del tag, construye el paquete, sube los artefactos y
  adjunta los `.whl`/`.tar.gz` a un release en GitHub.

## Flujo de versionamiento

1. Actualiza la versión en `pyproject.toml` del paquete deseado.
2. Crea el tag correspondiente, por ejemplo:

   ```bash
   git tag data-preparation-v0.1.0
   git push origin data-preparation-v0.1.0
   ```

3. El workflow `release` generará automáticamente el release con los
   artefactos listos para descargarse desde GitHub.
