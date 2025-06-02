# En configuraciones.py (o run_aco_from_snapshot.py)

import json
import numpy as np
import pandas as pd # ¡Asegúrate de importar pandas!
import time         # ¡Asegúrate de importar time!
import llbaco_aux 

def main():
    snapshot_file = 'network_snapshot_20250530_094552.json' # Tu archivo JSON

    try:
        with open(snapshot_file, 'r') as f:
            snapshot_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: El archivo '{snapshot_file}' no se encontró.")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo '{snapshot_file}' no es un JSON válido.")
        return

    # Extraer los nodos y enlaces directamente del JSON
    nodes = snapshot_data.get('switches', [])
    links_from_json = snapshot_data.get('links', [])
    
    # Puedes obtener el 'counter' del JSON si existe, de lo contrario usa un valor por defecto
    snapshot_counter = snapshot_data.get('counter', 1) 

    if not nodes or not links_from_json:
        print("Error: El JSON no contiene 'switches' o 'links' válidos.")
        return

    print(f"Nodos cargados: {nodes}")
    print(f"Número de enlaces cargados: {len(links_from_json)}")

    # Define los parámetros de ACO (ajústalos según tus necesidades)
    src_node_dpid = 1  # Por ejemplo, el nodo origen
    dst_node_dpid = 15 # Por ejemplo, el nodo destino
    delta_param = 0.5   # Parámetro para la función de costo

    # Parámetros para run_aco_llbaco
    aco_params = {
        'colony_size': 20,
        'iterations': 200,
        'alpha': 1.0,
        'beta': 1.0, 
        'gamma': 1.0,
        'rho': 0.05,
        'Q': 1.0,
        'high_cost': 1000,
        'q0': 0.5,
        'phi': 0.95
    }

    # --- Construir las matrices de costo y carga (esto se hace solo una vez) ---
    cost_matrix, load_matrix = llbaco_aux.build_cost_load_matrix2(
        nodes,
        links_from_json,
        delta=delta_param,
        high_cost=aco_params['high_cost']
    )

    if cost_matrix is None or load_matrix is None:
        print("No se pudieron construir las matrices de costo/carga. Saliendo.")
        return

    print("\nMatriz de Costos:")
    print(cost_matrix)
    print("\nMatriz de Cargas:")
    print(load_matrix)

    # --- Lógica para múltiples corridas y guardado a CSV ---
    experiment_runs = 30 # Define cuántas veces quieres ejecutar el ACO

    print(f"\n=== Experimento: ejecutando {experiment_runs} corridas de ACO para snapshot {snapshot_counter} ===")
    
    costs = []
    paths = []
    times = [] # Para almacenar el tiempo de cada corrida

    for i in range(experiment_runs):
        start_time_aco = time.time()
        
        # Llama directamente a llbaco_aux.run_aco_llbaco con las matrices ya construidas
        path_i, cost_i = llbaco_aux.run_aco_llbaco(
            nodes,
            cost_matrix,
            load_matrix,
            src_node_dpid,
            dst_node_dpid,
            **aco_params
        )
        end_time_aco = time.time()
        time_aco_run = end_time_aco - start_time_aco

        costs.append(cost_i)
        paths.append(" → ".join(map(str, path_i)))
        times.append(time_aco_run) # Guarda el tiempo de esta corrida
        print(f"Run {i+1:2d}/{experiment_runs}: cost={cost_i:.6f}, path={' → '.join(map(str, path_i))}, time={time_aco_run:.6f} s")

    # Calcular estadísticas después de todas las corridas
    mean_cost = float(np.mean(costs))
    std_cost  = float(np.std(costs, ddof=1)) # ddof=1 para desviación estándar de muestra
    total_aco_time = sum(times) # Suma de todos los tiempos de corrida

    print(f"\n--- Resumen del Experimento para snapshot {snapshot_counter} ---")
    print(f"Costo promedio: {mean_cost:.6f}")
    print(f"Desviación estándar del costo: {std_cost:.6f}")
    print(f"Tiempo total de ejecución de ACO ({experiment_runs} corridas): {total_aco_time:.6f} s")
    print(f"Tiempo promedio por corrida: {total_aco_time / experiment_runs:.6f} s")

    # --- Guardar a archivo CSV ---
    df = pd.DataFrame({
        'snapshot': [snapshot_counter] * experiment_runs,
        'run': list(range(1, experiment_runs + 1)),
        'cost': costs,
        'path': paths,
        'time_s': times # Añadido el tiempo de cada corrida
    })
    df['mean_cost'] = mean_cost
    df['std_cost'] = std_cost
    df['total_aco_time_s'] = total_aco_time
    
    # Nombre del archivo CSV (usando el contador del snapshot)
    filename = f"conf_{snapshot_counter}.csv"
    df.to_csv(filename, index=False)
    print(f"Resultados del experimento guardados en: {filename}")

if __name__ == "__main__":
    main()