"""
Esse módulo tem uma única função: carregar a base de dados Car Evaluation.

A ideia aqui é separar isso num arquivo próprio - assim, se um dia
precisarmos mudar de onde ou como os dados são lidos (por exemplo, trocar
de um CSV para um banco de dados), só mexemos aqui, sem bagunçar o resto
do projeto. Isso é o que chamam de "responsabilidade única" em Clean Code.
"""

import os
import pandas as pd


# Aqui a gente descobre o caminho absoluto da pasta onde este arquivo
# está guardado (a pasta src/), sobe um nível para chegar na raiz do
# projeto, e então entra na pasta data/. O motivo de fazer isso em vez
# de simplesmente escrever "data/car.data" é para o código funcionar
# sempre, não importa de qual pasta a gente rode o script no terminal.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "car.data")


def load_car_evaluation_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Lê o arquivo car.data e devolve os dados já num DataFrame do pandas,
    prontos para serem usados no resto do projeto.

    Args:
        path: caminho até o arquivo car.data (por padrão, já aponta
            para o arquivo dentro da pasta data/ do projeto).

    Returns:
        Um DataFrame com os 6 atributos do carro (buying, maint, doors,
        persons, lug_boot, safety) e a coluna class, que é o que
        queremos prever.
    """
    df = pd.read_csv(path)
    return df


def describe_dataset(df: pd.DataFrame) -> None:
    """
    Mostra no terminal um resumo rápido da base de dados: quantas
    instâncias e atributos existem, se falta algum valor, se tem
    linha duplicada, e como as classes estão distribuídas.

    A ideia é justamente responder, de forma automática, aquelas
    perguntas básicas que sempre fazemos antes de mexer em qualquer
    base de dados: "quantos dados eu tenho?", "está tudo completo?",
    "tem lixo duplicado no meio?".

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


# Esse bloco só roda quando executamos este arquivo diretamente
# (ex: "python src/data_loader.py"), e não quando ele é importado
# por outro módulo do projeto. É útil pra testar rapidinho se o
# carregamento dos dados está funcionando, sem depender do resto
# do pipeline.
if __name__ == "__main__":
    dataset = load_car_evaluation_data()
    describe_dataset(dataset)