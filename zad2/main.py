import numpy as np
from scipy.optimize import linprog

def read_matrix():
    print("Wprowadź macierz wypłat (oddziel elementy spacją, wiersze enterem). Pusta linia kończy:")
    matrix = []
    while True:
        line = input()
        if not line.strip():
            break
        row = list(map(float, line.strip().split()))
        matrix.append(row)
    return matrix

def minimax(matrix):
    row_mins = [min(row) for row in matrix]
    max_of_row_mins = max(row_mins)
    a_strategy = row_mins.index(max_of_row_mins)

    col_maxs = [max(col) for col in zip(*matrix)]
    min_of_col_maxs = min(col_maxs)
    b_strategy = col_maxs.index(min_of_col_maxs)

    print("\n--- Kryterium minimaksowe (Walda) ---")
    print(f"Minimalne wartości w wierszach gracza A: {row_mins}")
    print(f"Maksimum z tych minimów (strategia A): {max_of_row_mins}")
    print(f"Maksymalne wartości w kolumnach gracza B: {col_maxs}")
    print(f"Minimum z tych maksimów (strategia B): {min_of_col_maxs}")

    if max_of_row_mins == min_of_col_maxs:
        print("\nGra ma rozwiązanie w strategiach czystych!")
        print(f"Gracz A powinien wybrać strategię {a_strategy + 1}")
        print(f"Gracz B powinien wybrać strategię {b_strategy + 1}")
        print(f"Wartość gry: {max_of_row_mins}")
        return True
    else:
        print("\nGra nie ma rozwiązania w strategiach czystych. Przechodzimy dalej.")
        return False

def is_dominated(vec1, vec2, player='A'):
    if player == 'A':
        return all(a <= b for a, b in zip(vec1, vec2)) and any(a < b for a, b in zip(vec1, vec2))
    else:
        return all(a >= b for a, b in zip(vec1, vec2)) and any(a > b for a, b in zip(vec1, vec2))

def remove_dominated_strategies(matrix):
    matrix = np.array(matrix)
    rows_to_keep = []
    for i in range(len(matrix)):
        dominated = False
        for j in range(len(matrix)):
            if i != j and is_dominated(matrix[i], matrix[j], player='A'):
                dominated = True
                break
        if not dominated:
            rows_to_keep.append(i)
    matrix = matrix[rows_to_keep, :]

    cols_to_keep = []
    for i in range(matrix.shape[1]):
        dominated = False
        for j in range(matrix.shape[1]):
            if i != j and is_dominated(matrix[:, i], matrix[:, j], player='B'):
                dominated = True
                break
        if not dominated:
            cols_to_keep.append(i)
    matrix = matrix[:, cols_to_keep]

    print("\n--- Po usunięciu strategii zdominowanych ---")
    print(matrix)
    return matrix

def solve_lp_strategy(matrix, player='A'):
    matrix = np.array(matrix)
    m, n = matrix.shape

    if player == 'A':
        # Gracz A: zmienne to v oraz p_i (prawdopodobieństwa dla wierszy)
        # Maksymalizujemy v => minimalizujemy -v
        c = [-1] + [0] * m

        # Ograniczenia: suma (p_i * a_ij) >= v dla każdej kolumny j
        # Przekształcone: -p_i * a_ij <= -v
        A_ub = []
        b_ub = []
        for j in range(n):
            row = [1]  # współczynnik dla v
            # dla p_i jest -a_ij
            row.extend(-matrix[:, j])
            A_ub.append(row)
            b_ub.append(0)

        # suma p_i = 1
        A_eq = [[0] + [1] * m]
        b_eq = [1]

        # p_i >= 0, v dowolne
        bounds = [(None, None)] + [(0, 1) for _ in range(m)]

    else:
        # Gracz B: zmienne to v oraz q_j (prawdopodobieństwa dla kolumn)
        # Minimalizujemy v
        c = [1] + [0] * n

        # Ograniczenia: suma (q_j * a_ij) <= v dla każdej linii i
        # Przekształcone: p_j * a_ij - v <= 0
        A_ub = []
        b_ub = []
        for i in range(m):
            row = [-1]  # współczynnik dla v (teraz -v, bo v >= suma)
            row.extend(matrix[i, :])
            A_ub.append(row)
            b_ub.append(0)

        # suma q_j = 1
        A_eq = [[0] + [1] * n]
        b_eq = [1]

        # q_j >= 0, v dowolne
        bounds = [(None, None)] + [(0, 1) for _ in range(n)]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        v = res.x[0]
        probs = res.x[1:]
        print(f"\n--- Rozwiązanie strategii mieszanej (gracz {player}) ---")
        for i, p in enumerate(probs):
            print(f"Strategia {i + 1}: {round(p, 4)}")
        print(f"Wartość gry: {round(v, 4)}")
        return v, probs
    else:
        print(f"Nie udało się znaleźć rozwiązania programowania liniowego dla gracza {player}.")
        return None, None

if __name__ == "__main__":
    matrix = read_matrix()
    if not minimax(matrix):
        matrix = remove_dominated_strategies(matrix)
        solve_lp_strategy(matrix, player='A')
        solve_lp_strategy(matrix, player='B')
