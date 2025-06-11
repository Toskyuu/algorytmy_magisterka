import heapq
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os


def read_graph_from_csv(filename):
    try:
        df = pd.read_csv(filename, index_col=0)

        # Format: lista krawędzi
        if {'source', 'target', 'weight'}.issubset(df.columns):
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_edge(str(row['source']), str(row['target']), weight=float(row['weight']))

        # Format: macierz sąsiedztwa
        else:
            G = nx.Graph()
            for i in df.index:
                for j in df.columns:
                    if i == j:
                        continue  # pomijamy przekątną (połączenie samego ze sobą)
                    value = df.loc[i, j]

                    if pd.isna(value) or value == '' or value == 0:
                        continue  # brak połączenia

                    G.add_edge(str(i), str(j), weight=float(value))

        return G

    except Exception as e:
        print("Błąd podczas wczytywania pliku CSV:", e)
        return None

def compute_minimum_spanning_tree(G):
    mst = nx.minimum_spanning_tree(G, algorithm='prim')
    return mst

def prim_mst(G):
    start_node = list(G.nodes())[0]  # dowolny wierzchołek początkowy
    print(start_node)
    visited = set([start_node])
    edges = [
        (data['weight'], start_node, neighbor)
        for neighbor, data in G[start_node].items()
    ]
    heapq.heapify(edges)

    mst_edges = []

    while edges:
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            for neighbor, data in G[v].items():
                if neighbor not in visited:
                    heapq.heappush(edges, (data['weight'], v, neighbor))

    return mst_edges


def draw_graph(G, title):
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title(title)
    plt.show()

def main():
    filename = input("Podaj nazwę pliku CSV z grafem: ")
    if not os.path.exists(filename):
        print("Plik nie istnieje.")
        return

    G = read_graph_from_csv(filename)
    if G is None:
        return

    print("Graf został wczytany. Liczba wierzchołków:", G.number_of_nodes())
    draw_graph(G, "Oryginalny graf")

    mst_edges = prim_mst(G)

    # Tworzymy nowy graf MST do rysowania
    mst = nx.Graph()
    for u, v, weight in mst_edges:
        mst.add_edge(u, v, weight=weight)
    print("Minimalne drzewo rozpinające zawiera krawędzie:")
    for u, v, data in mst.edges(data=True):
        print(f"{u} - {v}, waga: {data['weight']}")

    draw_graph(mst, "Minimalne Drzewo Rozpinające (MST)")

if __name__ == "__main__":
    main()
