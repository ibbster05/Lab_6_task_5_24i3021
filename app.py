"""AI Lab - Interactive Search Visualization (GBFS / A*) using Streamlit + NetworkX.

Run locally:   streamlit run app.py
"""
import heapq
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# ---------------------------------------------------------------------------
# Graphs from the lab tasks (same coordinates / edges as in the notebook)
# ---------------------------------------------------------------------------
NETWORKS = {

    "Hospital Supply Robot (Task 3)": {
        "locations": {
            "Pharmacy": (0, 0), "Main_Corridor": (2, 1), "Patient_Wing": (1, 4),
            "Nursing_Station": (4, 2), "Laboratory": (5, 5), "Emergency_Ward": (8, 6),
        },
        "graph": {
            "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
            "Main_Corridor": {"Nursing_Station": 2.2},
            "Patient_Wing": {"Laboratory": 5.0},
            "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
            "Laboratory": {"Emergency_Ward": 3.2},
            "Emergency_Ward": {},
        },
        "start": "Pharmacy", "goal": "Emergency_Ward",
    },
}


# ---------------------------------------------------------------------------
# Search algorithms (same logic as the notebook)
# ---------------------------------------------------------------------------
def heuristic(locations, node, goal):
    (x1, y1), (x2, y2) = locations[node], locations[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _build_path(came_from, node):
    path = [node]
    while came_from[node] is not None:
        node = came_from[node]
        path.append(node)
    return path[::-1]


def greedy_best_first_search(start, goal, graph, locations):
    """f(n) = h(n)."""
    counter = 0
    frontier = [(heuristic(locations, start, goal), counter, start)]
    came_from, visited, order = {start: None}, set(), []
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        if current == goal:
            path = _build_path(came_from, goal)
            return path, sum(graph[a][b] for a, b in zip(path, path[1:])), order
        for nb in graph[current]:
            if nb not in visited and nb not in came_from:
                came_from[nb] = current
                counter += 1
                heapq.heappush(frontier, (heuristic(locations, nb, goal), counter, nb))
    return None, None, order


def a_star_search(start, goal, graph, locations):
    """f(n) = g(n) + h(n)."""
    counter = 0
    g = {start: 0.0}
    came_from, closed, order = {start: None}, set(), []
    frontier = [(heuristic(locations, start, goal), counter, start)]
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        order.append(current)
        if current == goal:
            return _build_path(came_from, goal), g[goal], order
        for nb, cost in graph[current].items():
            new_g = g[current] + cost
            if nb not in g or new_g < g[nb]:
                g[nb] = new_g
                came_from[nb] = current
                counter += 1
                heapq.heappush(frontier, (new_g + heuristic(locations, nb, goal), counter, nb))
    return None, None, order


ALGORITHMS = {"GBFS": greedy_best_first_search, "A*": a_star_search}


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_graph(graph, locations, path, title):
    G = nx.DiGraph()
    for node, nbrs in graph.items():
        G.add_node(node)
        for nb, cost in nbrs.items():
            G.add_edge(node, nb, weight=cost)

    path = path or []
    path_edges = list(zip(path, path[1:]))
    colors = []
    for n in G.nodes():
        if path and n == path[0]:
            colors.append("#7bd389")
        elif path and n == path[-1]:
            colors.append("#ffd166")
        elif n in path:
            colors.append("#ff9f68")
        else:
            colors.append("#a0c4ff")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    nx.draw_networkx_nodes(G, locations, node_color=colors, node_size=900, edgecolors="black", ax=ax)
    nx.draw_networkx_edges(G, locations, edge_color="gray", width=1.5, arrowsize=18, node_size=900, ax=ax)
    nx.draw_networkx_edges(G, locations, edgelist=path_edges, edge_color="red", width=4,
                           arrowsize=22, node_size=900, ax=ax)
    label_pos = {n: (x, y + 0.45) for n, (x, y) in locations.items()}
    nx.draw_networkx_labels(G, label_pos, font_size=9, font_weight="bold", ax=ax,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    nx.draw_networkx_edge_labels(G, locations, edge_labels=nx.get_edge_attributes(G, "weight"),
                                 font_size=9, label_pos=0.35, rotate=False, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    ax.margins(0.12)
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Informed Search Visualizer", layout="wide")
st.title("Informed Search Visualizer: GBFS vs A*")
st.caption("Pick a start node, a goal node and an algorithm, then run the search on the NetworkX graph.")

network_name = st.sidebar.selectbox("Network", list(NETWORKS))
net = NETWORKS[network_name]
locations, graph = net["locations"], net["graph"]
nodes = list(locations)

start = st.sidebar.selectbox("Initial node", nodes, index=nodes.index(net["start"]))
goal = st.sidebar.selectbox("Goal node", nodes, index=nodes.index(net["goal"]))
algo_name = st.sidebar.selectbox("Search algorithm", list(ALGORITHMS))
run = st.sidebar.button("Run search", type="primary")

if run:
    path, cost, order = ALGORITHMS[algo_name](start, goal, graph, locations)
    if path is None:
        st.error(f"No path exists from {start} to {goal} in this directed graph.")
        st.pyplot(draw_graph(graph, locations, [], f"{network_name}"))
    else:
        st.pyplot(draw_graph(graph, locations, path,
                             f"{algo_name}: {start} to {goal} | total cost = {cost:.2f}"))
        st.subheader("Result")
        st.markdown(f"**Selected algorithm:** {algo_name}")
        st.markdown(f"**Solution path:** {' → '.join(path)}")
        st.markdown(f"**Total path cost:** {cost:.2f}")
        st.markdown(f"**Node expansion order:** {' → '.join(order)}  ({len(order)} nodes)")
else:
    st.pyplot(draw_graph(graph, locations, [], network_name))
    st.info("Choose the options in the sidebar and press **Run search**.")
