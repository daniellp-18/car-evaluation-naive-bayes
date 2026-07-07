"""
Módulo responsável por carregar a base de dados Car Evaluation.

Este módulo isola a responsabilidade de leitura dos dados brutos,
seguindo o princípio de responsabilidade única (Clean Code).
"""

import os
import pandas as pd


# Aqui descobrimos o caminho absoluto da pasta onde este arquivo está
# (src/), subimos um nível para chegar na raiz do projeto, e então
# entramos na pasta data/. Isso garante que o carregamento funcione
# não importa de onde o script seja executado.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "car.data")


def load_car_evaluation_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega a base Car Evaluation a partir do arquivo CSV.

    Args:
        path: Caminho para o arquivo car.data.

    Returns:
        DataFrame contendo os 6 atributos preditores e a classe alvo.
    """
    df = pd.read_csv(path)
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """
    Exibe um resumo inicial da base de dados: dimensões, tipos de dados,
    valores ausentes e registros duplicados.

    Args:
        df: DataFrame já carregado com os dados.
    """
    print("=" * 50)
    print("RESUMO DA BASE DE DADOS - CAR EVALUATION")
    print("=" * 50)

    print(f"\nNúmero de instâncias: {df.shape[0]}")
    print(f"Número de atributos (colunas): {df.shape[1]}")

    print("\nColunas e tipos de dados:")
    print(df.dtypes)

    print("\nValores ausentes por coluna:")
    print(df.isnull().sum())

    n_duplicated = df.duplicated().sum()
    print(f"\nRegistros duplicados: {n_duplicated}")

    print("\nDistribuição da variável alvo (class):")
    print(df["class"].value_counts())

    print("\nPrimeiras 5 linhas:")
    print(df.head())


# Permite rodar este arquivo isoladamente para testar o carregamento
if __name__ == "__main__":
    dataset = load_car_evaluation_data()
    describe_dataset(dataset)
