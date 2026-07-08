"""
Módulo de avaliação do modelo Naive Bayes.

Chegou a hora da verdade: depois de treinar o modelo, precisamos saber
se ele realmente aprendeu alguma coisa útil ou só "decorou" os dados de
treino. Pra isso, testamos ele contra os dados de teste (que ele nunca
viu antes) e calculamos as métricas que mostram o desempenho de verdade:
Acurácia, Precisão, Recall, F1-Score e a Matriz de Confusão.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "results")
IMAGES_PATH = os.path.join(BASE_DIR, "images")


def evaluate_model(model, X_test, y_test):
    """
    Pede pro modelo prever as classes do conjunto de teste e depois
    compara essas previsões com as classes reais, calculando as
    principais métricas de desempenho.

    Args:
        model: o modelo já treinado (CategoricalNB).
        X_test: atributos do conjunto de teste.
        y_test: classes reais do conjunto de teste.

    Returns:
        Uma tupla (y_pred, accuracy, report), onde:
        - y_pred: o que o modelo previu para cada carro do teste;
        - accuracy: a acurácia geral, um número entre 0 e 1;
        - report: um texto pronto com Precisão, Recall e F1
          detalhados, classe por classe.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return y_pred, accuracy, report


def plot_confusion_matrix(y_test, y_pred, labels):
    """
    Monta e salva a imagem da matriz de confusão - uma tabela visual
    que mostra onde o modelo acertou (na diagonal principal) e, mais
    importante, onde ele errou e com qual classe confundiu.

    É um dos gráficos mais úteis pra entender o comportamento do
    modelo, porque a acurácia sozinha não conta essa história toda.

    Args:
        y_test: classes reais do conjunto de teste.
        y_pred: classes que o modelo previu.
        labels: lista com o nome das classes, na ordem que queremos
            que apareçam no gráfico.
    """
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matriz de Confusão - Naive Bayes (Car Evaluation)")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_PATH, "matriz_confusao.png"))
    plt.close()
    print("Gráfico salvo: matriz_confusao.png")


def save_metrics_to_file(accuracy, report):
    """
    Guarda a acurácia e o relatório completo de métricas num arquivo
    de texto dentro de results/, pra gente poder consultar depois sem
    precisar rodar o código de novo (e pra facilitar na hora de colar
    esses números no relatório ou nos slides).

    Args:
        accuracy: a acurácia geral do modelo.
        report: o texto gerado pelo classification_report.
    """
    caminho = os.path.join(RESULTS_PATH, "metrics.txt")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(f"Acurácia geral: {accuracy:.4f}\n\n")
        arquivo.write("Relatório de classificação (por classe):\n")
        arquivo.write(report)

    print(f"Métricas salvas em: {caminho}")


# Esse bloco permite rodar este arquivo sozinho, encadeando o
# pipeline inteiro (carregar -> pré-processar -> treinar -> avaliar)
# de uma vez só, sem precisar passar pelo main.py.
if __name__ == "__main__":
    from data_loader import load_car_evaluation_data
    from preprocessing import (
        encode_features,
        split_features_and_target,
        split_train_test,
    )
    from model import train_naive_bayes

    dataset = load_car_evaluation_data()
    X_encoded, encoder = encode_features(dataset)
    _, y = split_features_and_target(dataset)

    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)

    modelo = train_naive_bayes(X_train, y_train)

    y_pred, accuracy, report = evaluate_model(modelo, X_test, y_test)

    print("=" * 50)
    print("RESULTADOS DA AVALIAÇÃO")
    print("=" * 50)
    print(f"\nAcurácia geral: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("Relatório de classificação (por classe):")
    print(report)

    # Ordem fixa das classes, pra matriz de confusão sempre
    # aparecer organizada do mesmo jeito toda vez que rodarmos
    labels_ordenadas = ["unacc", "acc", "good", "vgood"]
    plot_confusion_matrix(y_test, y_pred, labels_ordenadas)

    save_metrics_to_file(accuracy, report)