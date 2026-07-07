"""
Módulo de avaliação do modelo Naive Bayes.

Aqui a gente testa o modelo já treinado contra os dados de teste
(que ele nunca viu antes) e calcula as métricas de desempenho:
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
    Faz as previsões no conjunto de teste e calcula as principais
    métricas de desempenho do modelo.

    Args:
        model: modelo já treinado (CategoricalNB).
        X_test: atributos do conjunto de teste.
        y_test: classes reais do conjunto de teste.

    Returns:
        Uma tupla (y_pred, accuracy, report), onde:
        - y_pred: as previsões que o modelo fez para o conjunto de teste;
        - accuracy: a acurácia geral (um número entre 0 e 1);
        - report: um texto com Precisão, Recall e F1 detalhados por classe.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return y_pred, accuracy, report


def plot_confusion_matrix(y_test, y_pred, labels):
    """
    Gera e salva a imagem da matriz de confusão, mostrando visualmente
    onde o modelo acertou (diagonal principal) e onde ele confundiu
    uma classe com outra.

    Args:
        y_test: classes reais do conjunto de teste.
        y_pred: classes previstas pelo modelo.
        labels: lista com o nome das classes, na ordem que queremos exibir.
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
    Salva a acurácia e o relatório de métricas em um arquivo de texto,
    dentro da pasta results/, para consulta posterior (e para colar
    no relatório final).

    Args:
        accuracy: acurácia geral do modelo.
        report: texto do classification_report.
    """
    caminho = os.path.join(RESULTS_PATH, "metrics.txt")

    with open(caminho, "w", encoding="utf-8") as arquivo:
        arquivo.write(f"Acurácia geral: {accuracy:.4f}\n\n")
        arquivo.write("Relatório de classificação (por classe):\n")
        arquivo.write(report)

    print(f"Métricas salvas em: {caminho}")


# Permite rodar este arquivo isoladamente, encadeando todo o pipeline
# (carregar -> pré-processar -> treinar -> avaliar) de uma vez só
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

    # Ordem fixa das classes, para a matriz de confusão sempre
    # aparecer organizada da mesma forma
    labels_ordenadas = ["unacc", "acc", "good", "vgood"]
    plot_confusion_matrix(y_test, y_pred, labels_ordenadas)

    save_metrics_to_file(accuracy, report)
