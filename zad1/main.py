import numpy as np


def wald_criterion(matrix, is_cost):
    if is_cost:
        max_values = np.max(matrix, axis=1)
        return np.argmin(max_values), max_values
    else:
        min_values = np.min(matrix, axis=1)
        return np.argmax(min_values), min_values


def optimistic_criterion(matrix, is_cost):
    if is_cost:
        min_values = np.min(matrix, axis=1)
        return np.argmin(min_values), min_values
    else:
        max_values = np.max(matrix, axis=1)
        return np.argmax(max_values), max_values


def hurwicz_criterion(matrix, alpha, is_cost):
    max_values = np.max(matrix, axis=1)
    min_values = np.min(matrix, axis=1)
    if is_cost:
        hurwicz_values = alpha * max_values + (1 - alpha) * min_values
        return np.argmin(hurwicz_values), hurwicz_values
    else:
        hurwicz_values = alpha * min_values + (1 - alpha) * max_values
        return np.argmax(hurwicz_values), hurwicz_values


def bayes_laplace_criterion(matrix, probabilities, is_cost):
    expected_values = matrix.dot(probabilities)
    if is_cost:
        return np.argmin(expected_values), expected_values
    else:
        return np.argmax(expected_values), expected_values


def savage_criterion(matrix, is_cost):
    if is_cost:
        min_per_column = np.min(matrix, axis=0)
        regret_matrix = matrix - min_per_column
    else:
        max_per_column = np.max(matrix, axis=0)
        regret_matrix = max_per_column - matrix

    print(regret_matrix)
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
    print("Czy dane to koszty (minimalizacja) czy zyski (maksymalizacja)? [k/z]:")
    data_type = input().strip().lower()
    is_cost = data_type == 'k'

    matrix = load_matrix()

    print("Podaj współczynnik ostrożności dla Hurwicza (0-1, np. 0.5) lub naciśnij Enter dla domyślnego 0.5:")
    alpha_input = input()
    alpha = float(alpha_input) if alpha_input.strip() else 0.5

    print("Podaj prawdopodobieństwa dla Bayesa (oddzielone spacjami, suma = 1) lub naciśnij Enter dla równych:")
    prob_input = input()
    if prob_input.strip():
        probabilities = list(map(float, prob_input.split()))
        if len(probabilities) != matrix.shape[1]:
            raise ValueError("Niepoprawna liczba prawdopodobieństw!")
        probabilities = np.array(probabilities)
    else:
        probabilities = np.full(matrix.shape[1], 1 / matrix.shape[1])

    print("\nMacierz danych:")
    print(matrix)

    wald_idx, wald_values = wald_criterion(matrix, is_cost)
    print(f"\nWald: wybrany wiersz {wald_idx + 1}, wartości: {wald_values}")

    opt_idx, opt_values = optimistic_criterion(matrix, is_cost)
    print(f"Optymistyczne: wybrany wiersz {opt_idx + 1}, wartości: {opt_values}")

    hurwicz_idx, hurwicz_values = hurwicz_criterion(matrix, alpha, is_cost)
    print(f"Hurwicz (alpha={alpha}): wybrany wiersz {hurwicz_idx + 1}, wartości: {hurwicz_values}")

    bayes_idx, bayes_values = bayes_laplace_criterion(matrix, probabilities, is_cost)
    print(f"Bayes-Laplace: wybrany wiersz {bayes_idx + 1}, wartości: {bayes_values}")

    savage_idx, savage_values = savage_criterion(matrix, is_cost)
    print(f"Savage: wybrany wiersz {savage_idx + 1}, wartości: {savage_values}")


if __name__ == "__main__":
    main()
