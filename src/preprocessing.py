"""
Módulo de pré-processamento da base Car Evaluation.

Aqui a gente prepara os dados pra entrar no modelo: transforma texto em
número (o CategoricalNB não entende "vhigh", "low" etc., só entende
números), separa quem é pergunta (X) de quem é resposta (y), e divide
tudo em treino e teste.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


# Mesma lista de sempre: os 6 atributos que descrevem o carro
FEATURE_COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# Aqui é o pulo do gato: em vez de deixar o OrdinalEncoder decidir
# a ordem das categorias sozinho (ele faria isso alfabeticamente,
# o que não faz sentido nenhum pra esses atributos), a gente define
# manualmente a ordem real de cada um, do "menor" pro "maior".
# Isso garante que o número codificado tenha significado de verdade:
# por exemplo, safety=low vira 0, med vira 1, high vira 2.
CATEGORY_ORDER = [
    ["low", "med", "high", "vhigh"],      # buying
    ["low", "med", "high", "vhigh"],      # maint
    ["2", "3", "4", "5more"],             # doors
    ["2", "4", "more"],                   # persons
    ["small", "med", "big"],              # lug_boot
    ["low", "med", "high"],               # safety
]

# Proporção reservada para teste (30%), seguindo a estratégia Holdout
# já usada nos slides do trabalho
TEST_SIZE = 0.3

# Fixamos essa semente só para garantir que, se alguém rodar o código
# de novo, a divisão treino/teste saia sempre igual (reprodutibilidade)
RANDOM_STATE = 42


def encode_features(df):
    """
    Transforma os 6 atributos categóricos (texto) em números inteiros,
    respeitando a ordem real de cada categoria (definida em
    CATEGORY_ORDER), e não uma ordem alfabética arbitrária.

    Args:
        df: DataFrame original, ainda com os atributos em texto.

    Returns:
        Uma tupla (X_encoded, encoder), onde X_encoded já está em
        números e encoder é o objeto treinado (guardamos ele porque
        pode ser útil depois, se quisermos decodificar algo).
    """
    X = df[FEATURE_COLUMNS]

    encoder = OrdinalEncoder(categories=CATEGORY_ORDER)
    X_encoded = encoder.fit_transform(X)

    return X_encoded, encoder


def split_features_and_target(df):
    """
    Separa o DataFrame em X (os 6 atributos) e y (a classe a prever).

    Args:
        df: DataFrame original com todas as colunas.

    Returns:
        Uma tupla (X, y).
    """
    X = df[FEATURE_COLUMNS]
    y = df["class"]

    return X, y


def split_train_test(X, y):
    """
    Divide os dados em treino e teste, usando a abordagem Holdout
    (70% treino / 30% teste), a mesma estratégia já apresentada
    nos slides do seminário.

    Args:
        X: atributos (já codificados ou não, tanto faz aqui).
        y: classe alvo.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # garante que a proporção das classes seja mantida
                     # tanto no treino quanto no teste - importante,
                     # já que sabemos que a base é desbalanceada
    )

    return X_train, X_test, y_train, y_test


# Permite rodar este arquivo isoladamente para conferir se está tudo certo
if __name__ == "__main__":
    from data_loader import load_car_evaluation_data

    dataset = load_car_evaluation_data()

    X, y = split_features_and_target(dataset)
    X_encoded, encoder = encode_features(dataset)

    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)

    print("Pré-processamento concluído com sucesso!\n")
    print(f"Total de instâncias: {len(dataset)}")
    print(f"Tamanho do treino: {len(X_train)}")
    print(f"Tamanho do teste: {len(X_test)}")

    print("\nPrimeiras 5 linhas de X já codificado:")
    print(X_encoded[:5])

    print("\nCategorias na ordem correta, por atributo:")
    for coluna, categorias in zip(FEATURE_COLUMNS, encoder.categories_):
        print(f"  {coluna}: {list(categorias)}")
