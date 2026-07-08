"""
main.py - Ponto de entrada do projeto Car Evaluation com Naive Bayes.

Este arquivo é o "maestro" do pipeline: ele chama, na ordem certa,
cada etapa que já construímos separadamente nos outros módulos.
A ideia é que, rodando só este arquivo, o projeto inteiro funcione
de ponta a ponta - do carregamento dos dados até a avaliação final
do modelo - sem precisar rodar nada manualmente etapa por etapa.
"""

from data_loader import load_car_evaluation_data, describe_dataset
from eda import (
    plot_class_distribution,
    plot_feature_frequencies,
    plot_class_by_safety,
)
from preprocessing import (
    encode_features,
    split_features_and_target,
    split_train_test,
)
from model import train_naive_bayes
from evaluation import evaluate_model, plot_confusion_matrix, save_metrics_to_file


def print_section(titulo):
    """
    Só uma função pequena para deixar o terminal mais organizado,
    imprimindo um cabeçalho bonito antes de cada etapa do pipeline.
    """
    print("\n" + "=" * 60)
    print(titulo)
    print("=" * 60)


def run_pipeline():
    """
    Executa o pipeline completo do projeto, do início ao fim:

    1. Carrega a base de dados;
    2. Gera os gráficos de análise exploratória (EDA);
    3. Faz o pré-processamento (encoding + separação treino/teste);
    4. Treina o modelo Naive Bayes;
    5. Avalia o modelo e salva os resultados.
    """

    # Etapa 1: carregando os dados brutos
    print_section("ETAPA 1/5 - CARREGANDO A BASE DE DADOS")
    dataset = load_car_evaluation_data()
    describe_dataset(dataset)

    # Etapa 2: análise exploratória, gerando os gráficos
    print_section("ETAPA 2/5 - ANÁLISE EXPLORATÓRIA (EDA)")
    plot_class_distribution(dataset)
    plot_feature_frequencies(dataset)
    plot_class_by_safety(dataset)

    # Etapa 3: pré-processamento dos dados
    print_section("ETAPA 3/5 - PRÉ-PROCESSAMENTO")
    X_encoded, encoder = encode_features(dataset)
    _, y = split_features_and_target(dataset)
    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)

    print(f"Total de instâncias: {len(dataset)}")
    print(f"Treino: {len(X_train)} carros | Teste: {len(X_test)} carros")

    # Etapa 4: treinamento do modelo
    print_section("ETAPA 4/5 - TREINAMENTO DO MODELO NAIVE BAYES")
    modelo = train_naive_bayes(X_train, y_train)
    print("Modelo CategoricalNB treinado com sucesso!")
    print(f"Classes aprendidas: {list(modelo.classes_)}")

    # Etapa 5: avaliação do modelo
    print_section("ETAPA 5/5 - AVALIAÇÃO DO MODELO")
    y_pred, accuracy, report = evaluate_model(modelo, X_test, y_test)

    print(f"\nAcurácia geral: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("Relatório de classificação por classe:")
    print(report)

    labels_ordenadas = ["unacc", "acc", "good", "vgood"]
    plot_confusion_matrix(y_test, y_pred, labels_ordenadas)
    save_metrics_to_file(accuracy, report)

    print_section("PIPELINE CONCLUÍDO COM SUCESSO")
    print("Todos os gráficos foram salvos em images/")
    print("As métricas completas foram salvas em results/metrics.txt")
    print("\nObrigado por rodar o projeto Car Evaluation - Naive Bayes!")


if __name__ == "__main__":
    run_pipeline()
