"""
Módulo de Análise Exploratória de Dados (EDA) da base Car Evaluation.

Antes de sair treinando qualquer modelo, vale a pena parar e "olhar"
pros dados de verdade - ver como as classes estão distribuídas, se
algum atributo chama atenção, se dá pra perceber algum padrão a olho
nu. É isso que esse módulo faz: gera gráficos e salva tudo na pasta
images/, prontos pra usar no relatório e nos slides da apresentação.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_car_evaluation_data


# Mesma ideia do data_loader.py: descobrir a raiz do projeto a partir
# de onde este arquivo está, pra garantir que o caminho das imagens
# funcione sempre, não importa de qual pasta a gente rode o script.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_PATH = os.path.join(BASE_DIR, "images")

# Os 6 atributos que descrevem o carro (deixamos de fora a coluna
# "class", já que essa é a resposta, não uma pergunta/atributo)
FEATURE_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


def plot_class_distribution(df):
    """
    Mostra, num gráfico de barras, quantos carros existem em cada
    categoria de aceitabilidade (unacc, acc, good, vgood).

    Esse é provavelmente o gráfico mais importante da EDA: ele deixa
    bem claro, só de olhar, que a base é desbalanceada - a grande
    maioria dos carros cai em "unacc", enquanto "good" e "vgood"
    aparecem bem menos. Isso vai explicar boa parte do comportamento
    do modelo mais pra frente.
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
    Gera um gráfico de barras pra cada um dos 6 atributos do carro,
    mostrando quantas vezes cada categoria aparece na base (por
    exemplo, quantos carros têm "buying = low", quantos têm "high",
    e assim por diante).

    Diferente da distribuição das classes, aqui a expectativa é ver
    os atributos bem mais equilibrados entre si - já que a base foi
    montada cobrindo todas as combinações possíveis de valores.
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
    Cruza o atributo "safety" com a classe final, pra ver se dá pra
    enxergar visualmente alguma relação entre segurança e
    aceitabilidade do carro.

    A intuição por trás disso é que segurança provavelmente pesa
    bastante na decisão - um carro inseguro dificilmente seria
    considerado "bom", não importa o resto dos atributos. Esse
    gráfico ajuda a confirmar (ou não) essa intuição antes mesmo
    de treinar qualquer modelo.
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


# Esse bloco só roda quando executamos este arquivo diretamente
# (ex: "python src/eda.py"), gerando os três gráficos de uma vez.
# Assim, se quisermos só atualizar as imagens sem rodar o pipeline
# inteiro de novo, basta rodar este arquivo sozinho.
if __name__ == "__main__":
    dataset = load_car_evaluation_data()

    plot_class_distribution(dataset)
    plot_feature_frequencies(dataset)
    plot_class_by_safety(dataset)

    print("\nAnálise exploratória concluída. Confira a pasta images/.")