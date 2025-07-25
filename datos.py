import pandas as pd
from pathlib import Path

def cargar_datos_crudos():
    base_path = Path("data")

    df_municipal = pd.read_excel(base_path / "Municipal.xlsx")
    df_departamental = pd.read_excel(base_path / "Departamental.xlsx")
    df_regional = pd.read_excel(base_path / "Regional.xlsx")

    if 'dato_municipio' in df_municipal.columns and 'dato_departamento' in df_municipal.columns:
        df_municipal['dato_municipio'] = df_municipal['dato_municipio'].fillna(df_municipal['dato_departamento'])

    return df_municipal, df_departamental, df_regional

