import numpy as np
import matplotlib.pyplot as plt
import random 


# ocnsiderra la maxima de cada uninad y dividrilo por ello
# llevar todo a valores de 0 y de 1
####
#def calculate_link_cost(load, delay, packet_loss, w_load=0.6, w_delay=0.2, w_packet_loss=0.2):
    # Evita división por cero; podrías sumar una pequeña constante
    #return w_load * (1/(load + 1e-6)) + w_delay * delay + w_packet_loss * packet_loss
####

def calculate_link_cost(delay, packet_loss, delta=0.5):
    """
    Calcula el costo de un enlace normalizando cada métrica.

    Parámetros:
      - delay: retraso medido en el enlace.
      - packet_loss: pérdida de paquetes en el enlace.
      - delta: valor de noramlizacion
      
    Devuelve:
      - El costo del enlace, con cada métrica normalizada en [0, 1].
    """
    delay = max(0, delay)
    packet_loss = max(0, packet_loss)
    return delta * delay + (1 - delta) * packet_loss


def build_cost_load_matrix(snapshot, nodes, topology_links, delta=0.5, high_cost=1000):
    """
    Construyes las matrices de coste y carga a partir de las métricas obtenidas.
    """
    n = len(nodes)
    # Inicializamos la matriz con high_cost
    cost_matrix = np.full((n, n), high_cost, dtype=float)
    load_matrix = np.zeros((n, n), dtype=float)

    
    for (src, dst, link_info) in topology_links:
        print(f"Procesando enlace: {src} -> {dst}, Puerto: {link_info.get('port')}")
        
        if src in snapshot and dst in snapshot:
            port = link_info.get('port')
            if port in snapshot[src]:
                metrics = snapshot[src][port]
                delay = metrics.get('delay', 0.0)
                packet_loss = metrics.get('packet_loss', 0.0)
                cost = calculate_link_cost(delay, packet_loss, delta)

                print(f"Costo calculado para {src} -> {dst}: {cost}")
                
                i = nodes.index(src)
                j = nodes.index(dst)
                cost_matrix[i, j] = cost
                cost_matrix[j, i] = cost  # Asumimos enlace simétrico


                # carga
                load = metrics.get('load', 0.0)
                print(f"Carga calculado para {src} -> {dst}: {load}")

                i = nodes.index(src)
                j = nodes.index(dst)
                load_matrix[i, j] = load
                load_matrix[j, i] = load  

            else:
                print(f"Puerto {port} no encontrado en el nodo {src}")
                i = nodes.index(src)
                j = nodes.index(dst)
                cost_matrix[i, j] = high_cost
                cost_matrix[j, i] = high_cost
                
        else:
            print(f"Nodo {src} o {dst} no encontrado en el snapshot")
            if src in nodes and dst in nodes:
                i = nodes.index(src)
                j = nodes.index(dst)
                cost_matrix[i, j] = high_cost
                cost_matrix[j, i] = high_cost
                
    # Si aún hay valores infinitos, se reemplazan por high_cost
    cost_matrix[np.isinf(cost_matrix)] = high_cost
    return cost_matrix, load_matrix

def build_cost_load_matrix2(
    nodes: list,             # Lista de IDs de switches (DPIDs) como en tu JSON "switches"
    links_data_json: list,   # La lista de diccionarios de enlaces de tu JSON "links"
    delta: float = 0.5,      # Parámetro delta para la función de costo
    high_cost: float = 1e9   # Un valor alto para representar enlaces no existentes o muy costosos
):
    """
    Construye las matrices de costo y carga a partir de la lista de enlaces del JSON.

    Args:
        nodes (list): Lista de IDs de switches (DPIDs).
        links_data_json (list): Lista de diccionarios de enlaces, cada uno con 'src', 'dst', 'load', 'delay', 'packet_loss'.
        delta (float): Parámetro para la función de costo (ej. para ponderar la carga).
        high_cost (float): Valor para enlaces inexistentes o inaccesibles.

    Returns:
        tuple: (cost_matrix, load_matrix) o (None, None) si hay un error.
    """
    if not nodes or not links_data_json:
        print("Advertencia: No se proporcionaron nodos o enlaces para construir las matrices.")
        return None, None

    num_nodes = len(nodes)
    # Mapeo de DPID a índice de matriz para acceso rápido
    node_to_idx = {node: i for i, node in enumerate(nodes)} 

    # Inicializar la matriz con high_cost para enlaces no definidos
    cost_matrix = np.full((num_nodes, num_nodes), high_cost, dtype=float)
    load_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    # La diagonal a 0.0 (costo de un nodo a sí mismo)
    np.fill_diagonal(cost_matrix, 0.0) 
    np.fill_diagonal(load_matrix, 0.0)

    for link_entry in links_data_json:
        src_dpid = link_entry.get('src')
        dst_dpid = link_entry.get('dst')
        load = link_entry.get('load', 0.0)
        delay = link_entry.get('delay', 0.0)
        packet_loss = link_entry.get('packet_loss', 0.0)

        # Asegurarse de que src y dst son válidos y están en la lista de nodos
        if src_dpid not in node_to_idx or dst_dpid not in node_to_idx:
            # print(f"Advertencia: Enlace ({src_dpid}-{dst_dpid}) en JSON no tiene nodos válidos en la lista de nodos. Saltando.")
            continue # Salta este enlace si los nodos no están en la topología

        src_idx = node_to_idx[src_dpid]
        dst_idx = node_to_idx[dst_dpid]

        # Calcular el costo usando la nueva función auxiliar
        cost = calculate_link_cost(delay, packet_loss, delta)
        
        # Asignar a las matrices
        cost_matrix[src_idx, dst_idx] = cost
        load_matrix[src_idx, dst_idx] = load

    return cost_matrix, load_matrix

def calculate_heuristic_matrices(load_matrix, cost_matrix, high_cost=1000):
    """
    Calcula las matrices de heurŕistica (eta = 1/Load y mu = 1/Cost)
    basándose en las matrices de carga y costo.
    Maneja divisiones por cero e infinitos.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_matrix = 1 / load_matrix
        eta_matrix[np.isinf(eta_matrix)] = high_cost * 10 
        eta_matrix[cost_matrix >= high_cost] = 1e-9 


        mu_matrix = 1 / cost_matrix

        mu_matrix[np.isinf(mu_matrix)] = high_cost * 10 # Un valor grande para favorecer caminos con bajo costo/costo cero
        
        # Los enlaces que no existen (cost_matrix = high_cost) deben tener mu 0 o muy pequeño.
        mu_matrix[cost_matrix >= high_cost] = 1e-9 # Valor muy pequeño para enlaces inaccesibles

    # Asegurarse de que las heurísticas sean positivas y finitas, manejando NaN (Not a Number)
    eta_matrix = np.nan_to_num(eta_matrix, nan=1e-9, posinf=high_cost*10, neginf=1e-9)
    mu_matrix = np.nan_to_num(mu_matrix, nan=1e-9, posinf=high_cost*10, neginf=1e-9)

    return eta_matrix, mu_matrix
    

    
def initialize_pheromones_matrix(nodes):
    """
    Inicializa la matriz de feromonas .
    """
    n = len(nodes)
    pheromones = np.ones((n, n))
    return pheromones

def calculate_path_load(path_indices, load_matrix, high_cost=1000):
    """
    Calcula la carga de un camino como la carga máxima en sus enlaces (Ecuación 5).
    """
    max_load = 0.0
    for i in range(len(path_indices) - 1):
        u = path_indices[i]
        v = path_indices[i+1]

        # Para asegursnos de que el enlace existe y no tiene una carga inaccesible
        # Utilizamos la matriz de cargas

        if load_matrix[u, v] < high_cost: # Evitar enlaces "inaccesibles"
             max_load = max(max_load, load_matrix[u, v])
        else:
             # Si un enlace en el path es inaccesible, la carga del path es 'inf' o high
             return float('inf') 
    return max_load

def calculate_path_cost_llbaco(path_indices, cost_matrix, high_cost=1000):
    """
    Calcula el costo de un camino como la suma de los costos de sus enlaces (Ecuación 6).
    Ignora enlaces con costo igual a high_cost.
    """
    total_cost = 0.0
    for i in range(len(path_indices) - 1):
        u = path_indices[i]
        v = path_indices[i+1]
        # Asegurarse de que el enlace existe y no tiene costo alto
        if cost_matrix[u, v] < high_cost:
            total_cost += cost_matrix[u, v]
        else:
            # Si un enlace en el path es inaccesible, el costo total es 'inf' o high_cost
            return float('inf') # Esto es mejor que sumarle high_cost

    return total_cost

def select_best_path_llbaco(paths_data_this_iteration, Lk, high_cost=1000):
    """
    Selecciona el mejor camino de la iteración basándose en el umbral de carga y mínimo costo.
    paths_data_this_iteration es una lista de (path_indices, path_cost, path_load).
    """
    # 1. Filtrar caminos con carga > Lk
    filtered_paths_data = [
        (path_indices, path_cost, path_load) for path_indices, path_cost, path_load in paths_data_this_iteration
        if path_load <= Lk and path_cost < float('inf') # Descartamos camino no válidos
    ]

    if not filtered_paths_data:
        return None, float('inf') # No hay ningún camino válido en esta iteración

    # 2. Seleccionar el camino con mínimo costo entre los filtrados (Ecuación 8)
    best_path_data_this_iteration = min(filtered_paths_data, key=lambda item: item[1]) # item[1] es path_cost

    return best_path_data_this_iteration[0], best_path_data_this_iteration[1] # Retorna (path_indices, path_cost)

def update_pheromones_global(pheromones, best_path_global_indices, best_path_global_cost, rho, Q):
    """
    Actualiza las feromonas globalmente según el Ant Colony System (ACS).
    Solo el mejor camino global deposita feromona y solo en sus arcos hay evaporación global.     
    """

    # 1. Evaporación (Ecuación 3, parte (1-rho)*tau)
    pheromones *= (1 - rho)

    # 2. Depósito (Ecuación 3, parte +Delta_tau_ij y Ecuación 4)
    # Delta_tau es Q/L si (i,j) pertenece al mejor camino, 0 en otro caso.
    if best_path_global_indices is not None and best_path_global_cost > 0: # Asegurarse de tener un camino válido y costo > 0
        delta_tau = Q / best_path_global_cost
        for i in range(len(best_path_global_indices) - 1):
            u = best_path_global_indices[i]
            v = best_path_global_indices[i+1]

            # Evaporación y depósito en el mismo paso para los arcos del mejor camino global
            pheromones[u, v] = (1 - rho) * pheromones[u, v] + rho * delta_tau 
            pheromones[v, u] = (1 - rho) * pheromones[v, u] + rho * delta_tau 


def update_pheromones_local(pheromones, u, v, phi, tau0):
    """
    Actualiza las feromonas localmente del arco(u,v)
    Phi es la tasa de evaporación, tau0 es el valor inicial/reset
    """
    # Ecuación: tau_uv = (1-phi) * tau_uv + phi*tau0
    pheromones[u, v] = (1-phi) * pheromones[u,v] + phi*tau0
    pheromones[v, u] = (1-phi) * pheromones[u,v] + phi*tau0
 
def search_path_by_ant(src_idx, dst_idx, n_nodes,
                        pheromones, eta_matrix, mu_matrix, cost_matrix,
                        alpha, beta, gamma, high_cost, q0, phi, tau0):
    """
    Simula el movimiento de una sola hormiga buscando un camino
    desde src_idx hasta dst_idx usando la probabilidad de transición de LLBACO.

    Retorna la lista de ínidces del camino encontrado o una lista vacía si no se llega al destino.
    """
    current_node_idx = src_idx
    path_indices = [current_node_idx]
    visited = {current_node_idx} 
    max_path_length = n_nodes 

    while current_node_idx != dst_idx and len(path_indices) < max_path_length:
        # Nodos permitidos: no visitados y con enlace válido (costo < high_cost)
        allowed_nodes_idx = [
            i for i in range(n_nodes)
            if i != current_node_idx and i not in visited and cost_matrix[current_node_idx, i] < high_cost
        ]

        if not allowed_nodes_idx:
            break # Hormiga atascada: no hay nodos válidos a donde ir

        q = random.random()
        next_node_idx = -1

        if q <= q0: # elegir el mejor enlace (greedy)
            best_term_value = -1.0
            best_next_node = -1

            for next_node_idx in allowed_nodes_idx:
                tau = pheromones[current_node_idx, next_node_idx]
                eta = eta_matrix[current_node_idx, next_node_idx]
                mu = mu_matrix[current_node_idx, next_node_idx]

                # Asegurar positividad para potencias y manejar valores extremos/no finitos
                tau = max(tau, 1e-9)
                eta = max(eta, 1e-9)
                mu = max(mu, 1e-9)

                term = (tau ** alpha) * (eta ** beta) * (mu ** gamma)

                if not np.isfinite(term) or term < 1e-15:
                    term = 0

                if term > best_term_value:
                    best_term_value = term
                    best_next_node = next_node_idx
            if best_next_node != -1:
                next_node_idx = best_next_node
            else:
                pass 

        # Seleccionar el siguiente nodo usando la ruleta
        if next_node_idx == -1:
            # Calcular probabilidades de transición para los nodos permitidos (Ecuación 1)
            probabilities = []
            denominator = 0.0

            for next_node_idx in allowed_nodes_idx:
                tau = pheromones[current_node_idx, next_node_idx]
                eta = eta_matrix[current_node_idx, next_node_idx]
                mu = mu_matrix[current_node_idx, next_node_idx]

                # Asegurar positividad para potencias y manejar valores extremos/no finitos
                tau = max(tau, 1e-9)
                eta = max(eta, 1e-9)
                mu = max(mu, 1e-9)

                term = (tau ** alpha) * (eta ** beta) * (mu ** gamma)

                if not np.isfinite(term) or term < 1e-15:
                    term = 0

                probabilities.append(term)
                denominator += term

            if denominator > 1e-15:
                normalized_probabilities = np.array(probabilities) / denominator
                sum_probs = normalized_probabilities.sum()
                if sum_probs > 1e-12:
                    normalized_probabilities /= sum_probs
                else:
                    break # Suma casi cero, hormiga atascada

                try:
                    next_node_idx = np.random.choice(allowed_nodes_idx, p=normalized_probabilities)
                except ValueError as e:
                    break # Hormiga atascada

            else:
                break # Hormiga atascada (denominador cero)
        
        # Mover la hormiga y aplicamos la actualizció nlocal
        if next_node_idx != -1:

            #actualización local
            update_pheromones_local(pheromones, current_node_idx, next_node_idx, phi, tau0)

            current_node_idx = next_node_idx
            path_indices.append(current_node_idx)
            visited.add(current_node_idx)
        else:
             break # Hormiga atascada

    # --- Fin del movimiento de la hormiga ---

    # Devuelve el camino solo si se ha llegado al destino
    if path_indices[-1] == dst_idx:
        return path_indices
    else:
        return [] 


def run_aco_llbaco(nodes,cost_matrix,load_matrix, src_dpid, dst_dpid,
                   iterations=200, colony_size=100, alpha=1.0, beta=1.0, gamma=1.0,
                   rho=0.5, Q=1.0, high_cost=1000,  q0=0.9, phi=0.1):
    """
    Ejecuta el algoritmo LLBACO para encontrar una ruta de src_dpid a dst_dpid.
    Orquesta las diferentes etapas del algoritmo.
    """
    n_nodes = len(nodes)
    dpid_to_index = {dpid: i for i, dpid in enumerate(nodes)}
    src_idx = dpid_to_index.get(src_dpid)
    dst_idx = dpid_to_index.get(dst_dpid)

    if src_idx is None or dst_idx is None:
        print(f"Error: Source DPID {src_dpid} or Destination DPID {dst_dpid} not found in nodes list.")
        return [], float('inf')

    tau0 = 1.0 # Valor inicial de feromona (ACS)

    # Construir matrices necesarias
    pheromones = initialize_pheromones_matrix(nodes)

    eta_matrix, mu_matrix = calculate_heuristic_matrices(load_matrix, cost_matrix, high_cost) 

    best_path_global_indices = None
    best_cost_global = float('inf')

    for iter in range(iterations):
        paths_data_this_iteration = [] 

        # Cada hormiga busca un camino en esta iteración
        for ant in range(colony_size):
            # Ecuación 1
            path_indices = search_path_by_ant(
                src_idx, dst_idx, n_nodes,
                pheromones, eta_matrix, mu_matrix, cost_matrix,
                alpha, beta, gamma, high_cost,q0, phi, tau0
            )

            # Si la hormiga encuentra un camino al destino
            if path_indices:
                 path_load = calculate_path_load(path_indices, load_matrix, high_cost)
                 path_cost = calculate_path_cost_llbaco(path_indices, cost_matrix, high_cost)
                 # Solo añadir paths válidos (costo no infinito)
                 if path_cost < float('inf'):
                     paths_data_this_iteration.append((path_indices, path_cost, path_load))


        # --- Fin movimiento de hormigas en la iteración ---

        # Calcular Lk (Umbral de carga)
        if paths_data_this_iteration:
            total_load_this_iteration = sum(pd[2] for pd in paths_data_this_iteration)
            Lk = total_load_this_iteration / len(paths_data_this_iteration)
        else:
            Lk = float('inf') # Si no hay caminos, el umbral no filtra nada

        # Seleccionar el mejor camino de esta iteración (Ecuación 8)
        best_path_this_iteration_indices, best_cost_this_iteration = select_best_path_llbaco(
            paths_data_this_iteration, Lk
        )

        # Actualizar el mejor camino global
        if best_path_this_iteration_indices is not None and best_cost_this_iteration < best_cost_global:
             best_cost_global = best_cost_this_iteration
             best_path_global_indices = best_path_this_iteration_indices


        # Actualizar feromonas (global) (Ecuaciones 3 y 4)
        update_pheromones_global(pheromones, best_path_global_indices, best_cost_global, rho, Q)

    # --- Fin de las iteraciones ---

    # Convertir el mejor camino global de índices a DPIDs
    best_path_dpid = [nodes[i] for i in best_path_global_indices] if best_path_global_indices is not None else []

    return best_path_dpid, best_cost_global




# --------------------------------------------------
# Ejemplo de uso (prueba base)
# --------------------------------------------------

if __name__ == "__main__":
    # Datos de ejemplo: snapshot y topología
    snapshot_data = {
        1: {1: {'load': 0.01, 'delay': 0.02, 'packet_loss': 0.001}, 2: {'load': 0.05, 'delay': 0.08, 'packet_loss': 0.005}},
        2: {1: {'load': 0.01, 'delay': 0.02, 'packet_loss': 0.001}, 3: {'load': 0.03, 'delay': 0.06, 'packet_loss': 0.002}},
        3: {1: {'load': 0.03, 'delay': 0.06, 'packet_loss': 0.002}, 2: {'load': 0.04, 'delay': 0.07, 'packet_loss': 0.003}},
        4: {1: {'load': 0.05, 'delay': 0.08, 'packet_loss': 0.005}},
        5: {1: {'load': 0.02, 'delay': 0.05, 'packet_loss': 0.001}},
        6: {1: {'load': 0.01, 'delay': 0.03, 'packet_loss': 0.0005}},
        7: {1: {'load': 0.005, 'delay': 0.01, 'packet_loss': 0.0001}},
    }

    nodes_list = [1, 2, 3, 4, 5, 6, 7] 
    topology_links_list = [
        (1, 2, {'port': 1}), (1, 4, {'port': 2}),
        (2, 1, {'port': 1}), (2, 3, {'port': 3}),
        (3, 2, {'port': 1}),
        (4, 1, {'port': 1}), (4, 5, {'port': 4}),
        (5, 4, {'port': 1}), (5, 6, {'port': 2}),
        (6, 5, {'port': 1}), (6, 7, {'port': 3}),
        (7, 6, {'port': 1})
    ]

    src_node_dpid = 1 # Origen
    dst_node_dpid = 7 # Destino

    print(f"Ejecutando LLBACO para ruta de {src_node_dpid} a {dst_node_dpid}...")

    # Construir matrices de costo y carga
    cost_matrix, load_matrix = build_cost_load_matrix(
        snapshot_data, nodes_list, topology_links_list, delta=0.5, high_cost=1000
    )

    print("\n--- Matrices Construidas ---")
    print("Nodos (DPID <-> índice):")
    dpid_to_index = {dpid: i for i, dpid in enumerate(nodes_list)}
    index_to_dpid = {i: dpid for dpid, i in dpid_to_index.items()}
    print(index_to_dpid)
    print("\nCost Matrix (LLBACO):")
    print(cost_matrix)
    print("\nLoad Matrix:")
    print(load_matrix)
    print("--------------------------\n")


    best_path_dpids, best_cost = run_aco_llbaco(
        nodes_list,
        cost_matrix, 
        load_matrix,
        src_node_dpid,
        dst_node_dpid,
        iterations=200,
        colony_size=100,
        alpha=1.0, beta=2.0, gamma=2.0, 
        rho=0.1, 
        Q=100.0,
        high_cost=1000,
        q0=0.9,  
        phi=0.1  #
    )

    print("\n--- Resultado Final LLBACO (ACS-like) ---")
    print(f"Origen: {src_node_dpid}, Destino: {dst_node_dpid}")
    if best_path_dpids:
        print(f"Mejor Ruta Encontrada (DPIDs): {best_path_dpids}")
        print(f"Costo Asociado: {best_cost:.4f}")
    else:
        print("No se encuentra una ruta válida al destino.")