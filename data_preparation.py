"""Compatibility wrapper for ``crispdm_data_preparation`` package."""

import local_dev  # noqa: F401

from crispdm_data_preparation import *  # noqa: F401,F403


if __name__ == "__main__":
    from crispdm_data_preparation import get_preprocessed_dataframe

    df_proc = get_preprocessed_dataframe()
    print("\n[main] Shape del DataFrame preprocesado:", df_proc.shape)
    print("\n[main] Primeras filas:")
    print(df_proc.head())
