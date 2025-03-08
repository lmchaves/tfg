import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Funciones de utilidad para ACO
# -------------------------------

def inverse_costs(cost_matrix):
    """
    Calcula la inversa de la matriz de costos.
    Se añade una pequeña constante para evitar división por cero.
    """
    with np.errstate(divide='ignore'):
        inv_costs = 1 / (cost_matrix + 1e-6)
    return inv_costs

def initialize_ants(n_ants, n_nodes):
    """
    Inicializa las posiciones iniciales (aleatorias) de las hormigas.
    """
    return np.random.randint(0, n_nodes, size=n_ants)

def move_ants_on_costs(pheromones, inv_costs, positions, alpha, beta, del_tau, n_nodes):
    """
    Mueve las hormigas a través de los nodos basándose en las probabilidades que
    combinan la influencia de las feromonas y la heurística (inversa del costo).
    """
    paths = np.full((len(positions), n_nodes), -1, dtype=int)
    paths[:, 0] = positions
    for step in range(1, n_nodes):
        for ant in range(len(positions)):
            current = paths[ant, step-1]
            # Probabilidad basada en feromonas y heurística
            probs = (pheromones[current] ** alpha) * (inv_costs[current] ** beta)
            # Excluir nodos ya visitados
            visited = paths[ant, :step]
            probs[visited] = 0
            total = probs.sum()
            if total == 0:
                # Si ya visitó todos, elegir aleatoriamente entre los no visitados
                candidates = [i for i in range(n_nodes) if i not in visited]
                next_node = np.random.choice(candidates)
            else:
                probs = probs / total
                next_node = np.random.choice(n_nodes, p=probs)
            paths[ant, step] = next_node
            pheromones[current, next_node] += del_tau
    return paths

def run_aco_llbaco(cost_matrix, iterations=200, colony=100, alpha=2.0, beta=1.5, del_tau=1.0, rho=0.5):
    """
    Ejecuta el algoritmo ACO para optimizar una ruta (o ciclo) en la red,
    utilizando una matriz de costos derivada de las métricas de la red.
    """
    n = cost_matrix.shape[0]
    inv_costs = inverse_costs(cost_matrix)
    pheromones = np.ones((n, n))
    best_path = None
    best_cost = float('inf')
    for _ in range(iterations):
        positions = initialize_ants(colony, n)
        paths = move_ants_on_costs(pheromones, inv_costs, positions, alpha, beta, del_tau, n)
        # Evaporación de feromonas
        pheromones *= (1 - rho)
        for path in paths:
            # Se suma el costo de cada enlace en el camino
            cost = sum(cost_matrix[path[i], path[i+1]] for i in range(n-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path
    return best_path, best_cost

# --------------------------------------------------
# Funciones para calcular el costo de un enlace
# --------------------------------------------------

def calculate_link_cost(delay, packet_loss, delta=0.5):
    """
    Calcula el costo de un enlace según la fórmula:
    
      Cost = delta * delay + (1 - delta) * packet_loss
    
    Puedes ajustar delta para darle mayor peso al delay o a la pérdida de paquetes.
    """
    return delta * delay + (1 - delta) * packet_loss

def build_cost_matrix(snapshot, nodes, topology_links, delta=0.5, high_cost=1000):
    """
    Construye una matriz de costos a partir de las métricas de la red.

    Parámetros:
      - snapshot: diccionario con métricas por nodo y puerto, por ejemplo:
          { nodo: { puerto: {'load': ..., 'delay': ..., 'packet_loss': ...}, ... }, ... }
      - nodes: lista ordenada de IDs de nodos (por ejemplo, switches) presentes.
      - topology_links: lista de tuplas (src, dst, {'port': ...}) que representan enlaces en la red.
      - delta: parámetro para ponderar el delay en la fórmula de costo.
      - high_cost: costo asignado a enlaces sin datos (valor alto finito).
    
    Devuelve una matriz de costos (n x n). Si no se tienen datos para un enlace, se asigna high_cost.
    """
    n = len(nodes)
    # Inicializamos la matriz con high_cost
    cost_matrix = np.full((n, n), high_cost, dtype=float)
    
    for (src, dst, link_info) in topology_links:
        # Si ambos nodos tienen datos en el snapshot...
        if src in snapshot and dst in snapshot:
            port = link_info.get('port')
            # Si el puerto especificado existe en los datos de src:
            if port in snapshot[src]:
                metrics = snapshot[src][port]
                delay = metrics.get('delay', 0.0)
                packet_loss = metrics.get('packet_loss', 0.0)
                cost = calculate_link_cost(delay, packet_loss, delta)
                i = nodes.index(src)
                j = nodes.index(dst)
                cost_matrix[i, j] = cost
                cost_matrix[j, i] = cost  # Asumimos enlace simétrico
            else:
                # Si no existe el puerto, dejamos high_cost
                i = nodes.index(src)
                j = nodes.index(dst)
                cost_matrix[i, j] = high_cost
                cost_matrix[j, i] = high_cost
        else:
            # Si alguno de los nodos no está en el snapshot, se asigna high_cost
            if src in nodes and dst in nodes:
                i = nodes.index(src)
                j = nodes.index(dst)
                cost_matrix[i, j] = high_cost
                cost_matrix[j, i] = high_cost
                
    # Si aún hay valores infinitos, se reemplazan por high_cost
    cost_matrix[np.isinf(cost_matrix)] = high_cost
    return cost_matrix

# --------------------------------------------------
# Ejemplo de uso (prueba base)
# --------------------------------------------------

if __name__ == "__main__":
    # Datos de ejemplo: snapshot y topología según la información simulada o recogida por el monitor
    snapshot = {
        1: {1: {'load': 0.001, 'delay': 0.05, 'packet_loss': 0.0, 'connected_to': 2}},
        2: {1: {'load': 0.002, 'delay': 0.1, 'packet_loss': 0.0, 'connected_to': 1}},
        3: {1: {'load': 0.0015, 'delay': 0.2, 'packet_loss': 0.0, 'connected_to': 4}},
        4: {1: {'load': 0.001, 'delay': 0.15, 'packet_loss': 0.0, 'connected_to': 3}},
        5: {1: {'load': 0.002, 'delay': 0.3, 'packet_loss': 0.0, 'connected_to': 6}},
        6: {1: {'load': 0.001, 'delay': 0.25, 'packet_loss': 0.0, 'connected_to': 5}},
    }
    # Lista de nodos (IDs de switches) en orden
    nodes = [1, 2, 3, 4, 5, 6]
    # Topología de enlaces: cada enlace es una tupla (src, dst, {'port': puerto en src})
    topology_links = [
        (1, 2, {'port': 1}),
        (1, 4, {'port': 1}),
        (2, 3, {'port': 1}),
        (2, 4, {'port': 1}),
        (3, 4, {'port': 1}),
        (3, 5, {'port': 1}),
        (3, 6, {'port': 1}),
        (4, 5, {'port': 1}),
        (4, 6, {'port': 1}),
        (5, 6, {'port': 1})
    ]
    delta = 0.5  # Peso para el delay en la fórmula de costo

    # Construir la matriz de costos
    cost_matrix = build_cost_matrix(snapshot, nodes, topology_links, delta)
    print("Cost Matrix:")
    print(cost_matrix)
    
    # Ejecutar el ACO para LLBACO usando la matriz de costos
    best_path, best_cost = run_aco_llbaco(cost_matrix, iterations=100, colony=50, alpha=1.0, beta=1.0, del_tau=1.0, rho=0.5)
    print("Best Path:", best_path)
    print("Best Cost:", best_cost)
    
    # Para visualizar (opcional)
    coords = np.random.rand(len(nodes), 2) * 100
    path_coords = coords[best_path]
    plt.plot(path_coords[:, 0], path_coords[:, 1], marker='o')
    plt.title(f"ACO LLBACO Solution (Cost: {best_cost:.2f})")
    plt.show()
