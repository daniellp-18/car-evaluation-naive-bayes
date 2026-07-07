"""
Módulo de Análise Exploratória de Dados (EDA) da base Car Evaluation.

A ideia aqui é simples: antes de treinar qualquer modelo, é importante
"conhecer" os dados de verdade. Este módulo gera gráficos que mostram
como as classes estão distribuídas e como cada atributo se comporta,
salvando as imagens na pasta images/ para usarmos no relatório e nos slides.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_car_evaluation_data


# Mesma lógica do data_loader.py: descobrir a raiz do projeto a partir
# da localização deste arquivo, para que o caminho funcione sempre,
# não importa de onde o script seja executado.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_PATH = os.path.join(BASE_DIR, "images")

# Lista com os 6 atributos preditores (sem contar a coluna "class")
FEATURE_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


def plot_class_distribution(df):
    """
    Gera um gráfico de barras mostrando quantos carros existem
    em cada categoria de aceitabilidade (unacc, acc, good, vgood).

    Esse gráfico é importante porque mostra logo de cara que a base
    é desbalanceada: a maioria dos carros é "unacc".
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="class", order=df["class"].value_counts().index)
    plt.title("Distribuição das classes (aceitabilidade dos carros)")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade de carros")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_PATH, "distribuicao_classes.png"))
    plt.close()
    print("Gráfico salvo: distribuicao_classes.png")


def plot_feature_frequencies(df):
    """
    Gera um gráfico de barras para cada um dos 6 atributos preditores,
    mostrando quantas vezes cada categoria aparece na base.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, coluna in enumerate(FEATURE_COLUMNS):
        sns.countplot(data=df, x=coluna, ax=axes[i])
        axes[i].set_title(f"Frequência do atributo: {coluna}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Quantidade")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_PATH, "frequencia_atributos.png"))
    plt.close()
    print("Gráfico salvo: frequencia_atributos.png")


def plot_class_by_safety(df):
    """
    Gera um gráfico cruzando o atributo 'safety' com a classe final,
    para ver se existe alguma relação visual entre segurança e
    aceitabilidade do carro.
    """
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="safety", hue="class")
    plt.title("Relação entre segurança (safety) e aceitabilidade do carro")
    plt.xlabel("Nível de segurança")
    plt.ylabel("Quantidade de carros")
    plt.legend(title="Classe")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_PATH, "safety_vs_class.png"))
    plt.close()
    print("Gráfico salvo: safety_vs_class.png")


# Permite rodar este arquivo isoladamente para gerar todos os gráficos de uma vez
if __name__ == "__main__":
    dataset = load_car_evaluation_data()

    plot_class_distribution(dataset)
    plot_feature_frequencies(dataset)
    plot_class_by_safety(dataset)

    print("\nAnálise exploratória concluída. Confira a pasta images/.")
