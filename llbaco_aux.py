import numpy as np
import matplotlib.pyplot as plt


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
    Construye una matriz de costos a partir de las métricas de la red.
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

def calculate_heuristic_matrices(load_matrix, cost_matrix, high_cost=1000):
    """
    Calcula las matrices de heur\u00EDstica (eta = 1/Load y mu = 1/Cost)
    bas\u00E1ndose en las matrices de carga y costo.
    Maneja divisiones por cero e infinitos.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        eta_matrix = 1 / load_matrix
        eta_matrix[np.isinf(eta_matrix)] = high_cost * 10 
        eta_matrix[cost_matrix >= high_cost] = 1e-9 


        mu_matrix = 1 / cost_matrix
        # Manejar infinitos que puedan surgir de 1/0. Reemplazar por un valor grande finito.
        mu_matrix[np.isinf(mu_matrix)] = high_cost * 10 # Un valor grande para favorecer caminos con bajo costo/costo cero
        # Los enlaces que no existen (cost_matrix = high_cost) deben tener mu 0 o muy peque\u00F1o.
        mu_matrix[cost_matrix >= high_cost] = 1e-9 # Valor muy peque\u00F1o para enlaces inaccesibles

    # Asegurarse de que las heur\u00EDsticas sean positivas y finitas, manejando NaN (Not a Number)
    eta_matrix = np.nan_to_num(eta_matrix, nan=1e-9, posinf=high_cost*10, neginf=1e-9)
    mu_matrix = np.nan_to_num(mu_matrix, nan=1e-9, posinf=high_cost*10, neginf=1e-9)

    return eta_matrix, mu_matrix
    

    
def initialize_pheromones_matrix(nodes):
    """
    Inicializa la matriz de feromonas (t\u00EDpicamente con 1.0).
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
        # Asegurarse de que el enlace existe y no tiene costo alto (inaccesible)
        # Usamos la matriz de carga que construimos. Si el valor es 0 o high_cost,
        # podr\u00EDamos querer tratarlo como un camino inv\u00E1lido o asignarle una carga alta.
        # Si el enlace es inaccesible (costo alto), la hormiga no deber\u00EDa tomarlo,
        # pero si por alguna raz\u00F3n est\u00E1 en el path, su carga es indefinida, o muy alta.
        # Asumiremos que los paths v\u00E1lidos solo contienen enlaces con carga <= 1.0 (normalizada).
        # Si la carga en la matriz es 0, el m\u00E1ximo ser\u00E1 al menos 0.
        # Consideramos solo enlaces con carga definida.
        if load_matrix[u, v] < high_cost: # Evitar enlaces "inaccesibles"
             max_load = max(max_load, load_matrix[u, v])
        else:
             # Si un enlace en el path es inaccesible, la carga del path es 'inf' o high
             return float('inf') # O high_cost, dependiendo de c\u00F3mo se trate

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
        if path_load <= Lk and path_cost < float('inf') # Tambi\u00E9n descartar caminos inv\u00E1lidos
    ]

    if not filtered_paths_data:
        return None, float('inf') # No se encontr\u00F3 ning\u00FAn camino v\u00E1lido en esta iteraci\u00F3n

    # 2. Seleccionar el camino con m\u00EDnimo costo entre los filtrados (Ecuaci\u00F3n 8)
    best_path_data_this_iteration = min(filtered_paths_data, key=lambda item: item[1]) # item[1] es path_cost

    return best_path_data_this_iteration[0], best_path_data_this_iteration[1] # Retorna (path_indices, path_cost)

def update_pheromones_llbaco(pheromones, best_path_global_indices, best_path_global_cost, rho, Q):
    """
    Actualiza las feromonas globalmente (Ecuaci\u00F3n 3), a\u00F1adiendo feromonas solo en el mejor camino global (Ecuaci\u00F3n 4).
    """
    # 1. Evaporaci\u00F3n (Ecuaci\u00F3n 3, parte (1-rho)*tau)
    pheromones *= (1 - rho)

    # 2. Dep\u00F3sito (Ecuaci\u00F3n 3, parte +Delta_tau_ij y Ecuaci\u00F3n 4)
    # Delta_tau es Q/L si (i,j) pertenece al mejor camino, 0 en otro caso.
    if best_path_global_indices is not None and best_path_global_cost > 0: # Asegurarse de tener un camino v\u00E1lido y costo > 0
        delta_tau = Q / best_path_global_cost
        for i in range(len(best_path_global_indices) - 1):
            u = best_path_global_indices[i]
            v = best_path_global_indices[i+1]
            pheromones[u, v] += delta_tau
            # Si asumimos enlaces sim\u00E9tricos para las feromonas, tambi\u00E9n actualizar v, u
            pheromones[v, u] += delta_tau # Esto depende de si el modelo de feromonas es direccional o no. El PDF no lo aclara para la actualizaci\u00F3n.


def search_path_by_ant(src_idx, dst_idx, n_nodes,
                        pheromones, eta_matrix, mu_matrix, cost_matrix,
                        alpha, beta, gamma, high_cost):
    """
    Simula el movimiento de una sola hormiga buscando un camino
    desde src_idx hasta dst_idx usando la probabilidad de transici\u00F3n de LLBACO.

    Retorna la lista de \u00EDndices del camino encontrado o una lista vac\u00EDa si no se lleg\u00F3 al destino.
    """
    current_node_idx = src_idx
    path_indices = [current_node_idx]
    visited = {current_node_idx} # Usar un set para b\u00FAsqueda r\u00E1pida
    max_path_length = n_nodes # Limitar longitud m\u00E1xima para evitar bucles infinitos

    while current_node_idx != dst_idx and len(path_indices) < max_path_length:
        # Nodos permitidos: no visitados y con enlace v\u00E1lido (costo < high_cost)
        allowed_nodes_idx = [
            i for i in range(n_nodes)
            if i != current_node_idx and i not in visited and cost_matrix[current_node_idx, i] < high_cost
        ]

        if not allowed_nodes_idx:
            break # Hormiga atascada: no hay nodos v\u00E1lidos a donde ir

        # Calcular probabilidades de transici\u00F3n para los nodos permitidos (Ecuaci\u00F3n 1)
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

        # Seleccionar el siguiente nodo usando la ruleta
        next_node_idx = -1
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
                 # print(f"Error en np.random.choice: {e}") # Depuraci\u00F3n
                 break # Hormiga atascada

        else:
            break # Hormiga atascada (denominador cero)

        # Mover la hormiga
        if next_node_idx != -1:
            current_node_idx = next_node_idx
            path_indices.append(current_node_idx)
            visited.add(current_node_idx)
        else:
             break # Hormiga atascada

    # --- Fin del movimiento de la hormiga ---

    # Retornar el camino solo si lleg\u00F3 al destino
    if path_indices[-1] == dst_idx:
        return path_indices
    else:
        return [] # Retorna lista vac\u00EDa si no lleg\u00F3 a destino


def run_aco_llbaco(nodes,cost_matrix,load_matrix, src_dpid, dst_dpid,
                   iterations=200, colony_size=100, alpha=1.0, beta=1.0, gamma=1.0,
                   rho=0.5, Q=1.0, high_cost=1000):
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

    # Construir matrices necesarias
    pheromones = initialize_pheromones_matrix(nodes)

    eta_matrix, mu_matrix = calculate_heuristic_matrices(load_matrix, cost_matrix, high_cost) 

    best_path_global_indices = None
    best_cost_global = float('inf')

    for iter in range(iterations):
        # print(f"--- Iteraci\u00F3n {iter + 1}/{iterations} ---") # Depuraci\u00F3n
        paths_data_this_iteration = [] # Almacenar (path_indices, path_cost, path_load) para esta iteraci\u00F3n

        # Cada hormiga busca un camino en esta iteraci\u00F3n
        for ant in range(colony_size):
            # Ecuación 1
            path_indices = search_path_by_ant(
                src_idx, dst_idx, n_nodes,
                pheromones, eta_matrix, mu_matrix, cost_matrix,
                alpha, beta, gamma, high_cost
            )

            # Si la hormiga encontr\u00F3 un camino al destino
            if path_indices:
                 path_load = calculate_path_load(path_indices, load_matrix, high_cost)
                 path_cost = calculate_path_cost_llbaco(path_indices, cost_matrix, high_cost)
                 # Solo a\u00F1adir paths v\u00E1lidos (costo no infinito)
                 if path_cost < float('inf'):
                     paths_data_this_iteration.append((path_indices, path_cost, path_load))


        # --- Fin movimiento de hormigas en la iteraci\u00F3n ---

        # Calcular Lk (Umbral de carga)
        if paths_data_this_iteration:
            total_load_this_iteration = sum(pd[2] for pd in paths_data_this_iteration)
            Lk = total_load_this_iteration / len(paths_data_this_iteration)
        else:
            Lk = float('inf') # Si no hay caminos, el umbral no filtra nada

        # Seleccionar el mejor camino de esta iteraci\u00F3n (Ecuaci\u00F3n 8)
        best_path_this_iteration_indices, best_cost_this_iteration = select_best_path_llbaco(
            paths_data_this_iteration, Lk
        )

        # Actualizar el mejor camino global
        if best_path_this_iteration_indices is not None and best_cost_this_iteration < best_cost_global:
             best_cost_global = best_cost_this_iteration
             best_path_global_indices = best_path_this_iteration_indices
             # print(f"Nuevo mejor camino global (Iter {iter+1}): Costo = {best_cost_global:.4f}, Path (indices): {best_path_global_indices}") # Depuraci\u00F3n


        # Actualizar feromonas (Ecuaciones 3 y 4)
        update_pheromones_llbaco(pheromones, best_path_global_indices, best_cost_global, rho, Q)

    # --- Fin de las iteraciones ---

    # Convertir el mejor camino global de \u00EDndices a DPIDs
    best_path_dpid = [nodes[i] for i in best_path_global_indices] if best_path_global_indices is not None else []

    return best_path_dpid, best_cost_global




# --------------------------------------------------
# Ejemplo de uso (prueba base)
# --------------------------------------------------

if __name__ == "__main__":
    # Datos de ejemplo: snapshot y topolog\u00EDa
    # Ajusta estos datos a tu topolog\u00EDa real y m\u00E9tricas simuladas/reales
    snapshot_data = {
        1: {1: {'load': 0.01, 'delay': 0.02, 'packet_loss': 0.001}, 2: {'load': 0.05, 'delay': 0.08, 'packet_loss': 0.005}},
        2: {1: {'load': 0.01, 'delay': 0.02, 'packet_loss': 0.001}, 3: {'load': 0.03, 'delay': 0.06, 'packet_loss': 0.002}},
        3: {1: {'load': 0.03, 'delay': 0.06, 'packet_loss': 0.002}, 2: {'load': 0.04, 'delay': 0.07, 'packet_loss': 0.003}},
        4: {1: {'load': 0.05, 'delay': 0.08, 'packet_loss': 0.005}},
        5: {1: {'load': 0.02, 'delay': 0.05, 'packet_loss': 0.001}},
        6: {1: {'load': 0.01, 'delay': 0.03, 'packet_loss': 0.0005}},
        7: {1: {'load': 0.005, 'delay': 0.01, 'packet_loss': 0.0001}},
        # A\u00F1ade m\u00E1s switches y sus m\u00E9tricas por puerto
    }

    nodes_list = [1, 2, 3, 4, 5, 6, 7] # Aseg\u00FArate de que todos los DPIDs est\u00E9n aqu\u00ED

    topology_links_list = [
        (1, 2, {'port': 1}), (1, 4, {'port': 2}),
        (2, 1, {'port': 1}), (2, 3, {'port': 3}),
        (3, 2, {'port': 1}),
        (4, 1, {'port': 1}), (4, 5, {'port': 4}), # Este puerto 4 en SW 4 debe existir en snapshot[4]
        (5, 4, {'port': 1}), (5, 6, {'port': 2}), # Este puerto 2 en SW 5 debe existir en snapshot[5]
        (6, 5, {'port': 1}), (6, 7, {'port': 3}), # Este puerto 3 en SW 6 debe existir en snapshot[6]
        (7, 6, {'port': 1})
        # A\u00F1ade todos los enlaces bidireccionales.
        # Verifica que los puertos en topology_links_list existan como claves en el snapshot correspondiente al src_dpid.
        # Si no existen, la funci\u00F3n build_cost_and_load_matrices asignar\u00E1 high_cost al costo y 0.0 a la carga.
    ]

    src_node_dpid = 1 # Origen
    dst_node_dpid = 7 # Destino

    print(f"Ejecutando LLBACO para ruta de {src_node_dpid} a {dst_node_dpid}...")

    # Construir matrices de costo y carga
    cost_matrix, load_matrix = build_cost_load_matrix(
        snapshot_data, nodes_list, topology_links_list, delta=0.5, high_cost=1000
    )

    # *** A\u00F1adir impresi\u00F3n de las matrices ***
    print("\n--- Matrices Construidas ---")
    print("Nodos (DPID <-> \u00CDndice):")
    dpid_to_index = {dpid: i for i, dpid in enumerate(nodes_list)}
    index_to_dpid = {i: dpid for dpid, i in dpid_to_index.items()}
    print(index_to_dpid)
    print("\nCost Matrix (LLBACO):")
    print(cost_matrix)
    print("\nLoad Matrix:")
    print(load_matrix)
    print("--------------------------\n")
    # ***************************************


    # Inicializar la matriz de feromonas
    pheromones = initialize_pheromones_matrix(nodes_list)

    # Calcular las matrices de heur\u00EDstica (eta y mu)
    eta_matrix, mu_matrix = calculate_heuristic_matrices(load_matrix, cost_matrix, high_cost=1000)

    # Ejecutar el algoritmo LLBACO
    best_path_dpids, best_cost = run_aco_llbaco(
        nodes_list,
        cost_matrix, 
        load_matrix,
        src_node_dpid,
        dst_node_dpid,
        iterations=200,
        colony_size=100,
        alpha=1.0, beta=2.0, gamma=2.0, # Ajusta estos par\u00E1metros
        rho=0.5,
        Q=100.0,
        high_cost=1000
    )

    print("\n--- Resultado Final LLBACO ---")
    print(f"Origen: {src_node_dpid}, Destino: {dst_node_dpid}")
    if best_path_dpids:
        print(f"Mejor Ruta Encontrada (DPIDs): {best_path_dpids}")
        print(f"Costo Asociado: {best_cost:.4f}")
    else:
        print("No se encontr\u00F3 una ruta v\u00E1lida al destino.")