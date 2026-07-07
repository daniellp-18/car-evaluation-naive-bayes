"""
Módulo responsável por treinar o modelo Naive Bayes.

Aqui usamos o CategoricalNB, a variante do Naive Bayes feita
especificamente para atributos categóricos - que é exatamente
o tipo de dado que temos na base Car Evaluation.
"""

from sklearn.naive_bayes import CategoricalNB


def train_naive_bayes(X_train, y_train):
    """
    Treina um classificador Naive Bayes categórico com os dados
    de treino.

    Durante o treino, o modelo calcula, para cada atributo e para
    cada classe, a frequência com que cada categoria aparece.
    É basicamente ele "aprendendo" as probabilidades que depois
    vai usar no Teorema de Bayes para classificar carros novos.

    Args:
        X_train: atributos de treino, já codificados numericamente.
        y_train: classes de treino (unacc, acc, good, vgood).

    Returns:
        O modelo já treinado, pronto para fazer previsões.
    """
    model = CategoricalNB()
    model.fit(X_train, y_train)

    return model


# Permite rodar este arquivo isoladamente para conferir se o treino
# está funcionando direitinho
if __name__ == "__main__":
    from data_loader import load_car_evaluation_data
    from preprocessing import (
        encode_features,
        split_features_and_target,
        split_train_test,
    )

    dataset = load_car_evaluation_data()
    X_encoded, encoder = encode_features(dataset)
    _, y = split_features_and_target(dataset)

    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)

    modelo = train_naive_bayes(X_train, y_train)

    print("Modelo treinado com sucesso!")
    print(f"Classes aprendidas pelo modelo: {list(modelo.classes_)}")

    # Só para conferir rapidamente: vamos prever a classe do
    # primeiro carro do conjunto de teste, e comparar com a
    # classe real que ele tem de verdade
    primeira_previsao = modelo.predict(X_test[:1])
    print(f"\nExemplo de previsão (1º carro do teste):")
    print(f"  Classe prevista: {primeira_previsao[0]}")
    print(f"  Classe real:     {y_test.iloc[0]}")
