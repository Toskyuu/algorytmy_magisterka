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

def solve_lp_strategy_for_player_a(matrix):
    matrix = np.array(matrix)
    m, n = matrix.shape

    # LP: max v  <=>  min -v
    # Dodajemy zmienną v, która zostanie zminimalizowana (czyli -v zmaksymalizowane)
    c = [-1] + [0] * m  # Zmienna v, potem prawdopodobieństwa p1..pm

    # Ograniczenia: suma(pi * a_ij) >= v  =>  -pi * a_ij <= -v
    A_ub = []
    b_ub = []
    for j in range(n):
        constraint = [1]  # dla v
        constraint += [-matrix[i][j] for i in range(m)]  # -a_ij * p_i
        A_ub.append(constraint)
        b_ub.append(0)

    # Suma p_i = 1
    A_eq = [[0] + [1] * m]
    b_eq = [1]

    # Ograniczenia: pi >= 0
    bounds = [(None, None)] + [(0, 1) for _ in range(m)]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        v = res.x[0]
        probs = res.x[1:]
        print("\n--- Rozwiązanie strategii mieszanej (gracz A) ---")
        for i, p in enumerate(probs):
            print(f"Strategia {i + 1}: {round(p, 4)}")
        print(f"Wartość gry: {round(v, 4)}")
    else:
        print("Nie udało się znaleźć rozwiązania programowania liniowego.")

if __name__ == "__main__":
    matrix = read_matrix()
    if not minimax(matrix):
        matrix = remove_dominated_strategies(matrix)
        solve_lp_strategy_for_player_a(matrix)
