import numpy as np


def wald_criterion(matrix):
    min_values = np.min(matrix, axis=1)
    return np.argmax(min_values), min_values


def optimistic_criterion(matrix):
    max_values = np.max(matrix, axis=1)
    return np.argmax(max_values), max_values


def hurwicz_criterion(matrix, alpha):
    max_values = np.max(matrix, axis=1)
    min_values = np.min(matrix, axis=1)
    hurwicz_values = alpha * max_values + (1 - alpha) * min_values
    return np.argmax(hurwicz_values), hurwicz_values


def bayes_laplace_criterion(matrix, probabilities):
    expected_values = matrix.dot(probabilities)
    return np.argmax(expected_values), expected_values


def savage_criterion(matrix):
    max_per_column = np.max(matrix, axis=0)
    regret_matrix = max_per_column - matrix
    max_regret_per_row = np.max(regret_matrix, axis=1)
    return np.argmin(max_regret_per_row), max_regret_per_row


def load_matrix():
    print("Podaj liczbę decyzji (wiersze):")
    rows = int(input())
    print("Podaj liczbę stanów natury (kolumny):")
    cols = int(input())

    matrix = []
    print(f"Wprowadzaj dane wierszami (oddzielone spacjami, {cols} liczb na wiersz):")
    for i in range(rows):
        row = list(map(float, input(f"Wiersz {i + 1}: ").split()))
        if len(row) != cols:
            raise ValueError("Niepoprawna liczba elementów w wierszu!")
        matrix.append(row)

    return np.array(matrix)


def main():
    matrix = load_matrix()

    print("Podaj współczynnik ostrożności dla Hurwicza (0-1, np. 0.5):")
    alpha = float(input())

    print("Podaj prawdopodobieństwa dla Bayesa (oddzielone spacjami, suma = 1):")
    probabilities = list(map(float, input().split()))
    if len(probabilities) != matrix.shape[1]:
        raise ValueError("Niepoprawna liczba prawdopodobieństw!")
    probabilities = np.array(probabilities)

    print("\nMacierz użyteczności:")
    print(matrix)

    wald_idx, wald_values = wald_criterion(matrix)
    print(f"\nWald: wybrany wiersz {wald_idx}, wartości: {wald_values}")

    opt_idx, opt_values = optimistic_criterion(matrix)
    print(f"Optymistyczne: wybrany wiersz {opt_idx}, wartości: {opt_values}")

    hurwicz_idx, hurwicz_values = hurwicz_criterion(matrix, alpha)
    print(f"Hurwicz (alpha={alpha}): wybrany wiersz {hurwicz_idx}, wartości: {hurwicz_values}")

    bayes_idx, bayes_values = bayes_laplace_criterion(matrix, probabilities)
    print(f"Bayes-Laplace: wybrany wiersz {bayes_idx}, wartości: {bayes_values}")

    savage_idx, savage_values = savage_criterion(matrix)
    print(f"Savage: wybrany wiersz {savage_idx}, wartości: {savage_values}")


if __name__ == "__main__":
    main()
