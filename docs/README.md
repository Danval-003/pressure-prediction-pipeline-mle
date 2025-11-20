# Documentación

Este directorio agrupa guías prácticas para las piezas clave del
proyecto:

- `crispdm_packages.md`: estructura del monorepo y cómo trabajar con los
  paquetes de cada etapa CRISP-DM.
- `data_and_artifacts.md`: flujo de datos (split de `train.csv`,
  directorios esperados, modelos y artefactos).
- `docker_pipeline.md`: imagen con el pipeline completo + Streamlit y su
  CI/CD asociado.
- `docker_serve_api.md`: imagen liviana que solo sirve el modelo vía
  FastAPI, con ejemplos de requests.

Revisa cada archivo según la tarea que necesites realizar (entrenar,
evaluar, desplegar, consumir). Estos documentos complementan al
`README.md` raíz y contienen instrucciones más detalladas.
