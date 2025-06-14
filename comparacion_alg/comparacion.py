import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

aco_filename = 'confiables/conf_5.csv'
dijkstra_filename = 'confiables/dijkstra_delay_conf_5.csv'
aco2_filename = 'confiables/aco2_conf_5.csv'

df_aco = None
df_dijkstra = None
df_aco2 = None

try:
    if os.path.exists(aco_filename):
        df_aco = pd.read_csv(aco_filename)
        print(f"Datos cargados de: {aco_filename}")
    else:
        print(f"Advertencia: El archivo '{aco_filename}' no fue encontrado. Verifique la ruta y el nombre.")

    if os.path.exists(dijkstra_filename):
        df_dijkstra = pd.read_csv(dijkstra_filename)
        print(f"\nDatos cargados de: {dijkstra_filename}")
    else:
        print(f"Advertencia: El archivo '{dijkstra_filename}' no fue encontrado. Verifique la ruta y el nombre.")

    if os.path.exists(aco2_filename):
        df_aco2 = pd.read_csv(aco2_filename)
        print(f"\nDatos cargados de: {aco2_filename}")
    else:
        print(f"Advertencia: El archivo '{aco2_filename}' no fue encontrado. Verifique la ruta y el nombre.")

except Exception as e:
    print(f"Error al cargar archivos CSV: {e}")
    print("Asegúrate de que los archivos estén en el mismo directorio que este script o proporciona la ruta completa.")

dataframes_to_combine = [df for df in [df_aco, df_dijkstra, df_aco2] if df is not None]

if dataframes_to_combine:
    df_combined = pd.concat(dataframes_to_combine, ignore_index=True)

    # Renombrar los algoritmos
    df_combined['algorithm'] = df_combined['algorithm'].replace({
        'ACO': 'LLBACS',
        'ACO2': 'LLBACO',
        'Dijkstra_Delay': 'Dijkstra'
    })

    print("\n--- DataFrame Combinado para Análisis ---")
    print(df_combined.head())

    snapshot_num = df_combined['snapshot'].iloc[0] if not df_combined['snapshot'].empty else 'N/A'

    summary_costs = df_combined.groupby('algorithm')['cost'].agg(['mean', 'std']).fillna(0)
    print("\n--- Resumen de costes por Algoritmo (Media y Desviación Estándar) ---")
    print(summary_costs)

    summary_time = df_combined.groupby('algorithm')['time'].mean()
    print("\n--- Tiempo Promedio de Ejecución por Algoritmo ---")
    print(summary_time)

    summary_lsd = df_combined.groupby('algorithm')['lsd_link_load'].mean()
    print("\n--- LSD (Desviación Estándar de Carga de Enlaces) por Algoritmo ---")
    print(summary_lsd)

    summary_network_delay = df_combined.groupby('algorithm')['mean_network_delay'].mean()
    print("\n--- Retraso Promedio de la RED (Estado Inicial) por Algoritmo ---")
    print(summary_network_delay)

    # Filtrado para gráficas específicas (LLBACS y LLBACO)
    df_filtered_aco = df_combined[df_combined['algorithm'].isin(['LLBACS', 'LLBACO'])]

    summary_route_delay = df_combined.groupby('algorithm')['route_delay'].mean()
    print("\n--- Retraso Promedio de la Ruta (Impacto del Algoritmo) por Algoritmo ---")
    print(summary_route_delay)

    summary_route_load = df_combined.groupby('algorithm')['route_load'].mean()
    print("\n--- Carga Promedio de la Ruta (Impacto del Algoritmo) por Algoritmo ---")
    print(summary_route_load)

    # GRÁFICA 1: Comparación de costes Promedio con Barras de Error
    plt.figure(figsize=(9, 6))
    summary_costs['mean'].plot(
        kind='bar',
        yerr=summary_costs['std'],
        capsize=5,
        color=['skyblue', 'lightcoral', 'lightgreen']
    )
    plt.title(f'Comparación de costes Promedio de Ruta (Snapshot {snapshot_num})')
    plt.xlabel('Algoritmo')
    plt.ylabel('coste Promedio de la Ruta')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 2: coste de Ruta por Ejecución
    plt.figure(figsize=(10, 6))
    for algorithm_name, group in df_combined.groupby('algorithm'):
        plt.plot(group['run'], group['cost'], marker='o', linestyle='-', label=f'{algorithm_name}')

    plt.title(f'coste de Ruta por Ejecución por Algoritmo (Snapshot {snapshot_num})')
    plt.xlabel('Número de Ejecución')
    plt.ylabel('coste de la Ruta')
    plt.legend(title='Algoritmo')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 3: Comparación del Tiempo de Ejecución Promedio
    plt.figure(figsize=(9, 6))
    summary_time.plot(
        kind='bar',
        color=['lightgreen', 'orange', 'skyblue']
    )
    plt.title(f'Tiempo de Ejecución Promedio por Algoritmo (Snapshot {snapshot_num})')
    plt.xlabel('Algoritmo')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 4: LSD (Desviación Estándar de la Carga de Enlaces)
    plt.figure(figsize=(9, 6))
    summary_lsd.plot(
        kind='bar',
        color=['purple', 'gold', 'coral']
    )
    plt.title(f'LSD (Desviación Estándar de la Carga de Enlaces) (Snapshot {snapshot_num})')
    plt.xlabel('Algoritmo')
    plt.ylabel('LSD')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 5: Retraso Promedio de la RUTA (Impacto del Algoritmo) - SOLO LLBACS y LLBACO
    plt.figure(figsize=(9, 6))
    # Aquí es donde usamos df_filtered_aco
    df_filtered_aco.groupby('algorithm')['route_delay'].mean().plot(
        kind='bar',
        color=['teal', 'salmon']
    )
    plt.title(f'Retraso Promedio de la RUTA (LLBACS vs LLBACO) (Snapshot {snapshot_num})')
    plt.xlabel('Algoritmo')
    plt.ylabel('Retraso Promedio de la Ruta (segundos/ms)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 6: Retraso Promedio de la RED (Estado Inicial)
    plt.figure(figsize=(9, 6))
    summary_network_delay.plot(
        kind='bar',
        color=['gray', 'darkblue', 'yellowgreen']
    )
    plt.title(f'Retraso Promedio de la RED (Estado Inicial) (Snapshot {snapshot_num})')
    plt.xlabel('Algoritmo')
    plt.ylabel('Retraso Promedio de la Red (segundos/ms)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # GRÁFICA 7: Carga Promedio de la RUTA (Impacto del Algoritmo) - SOLO LLBACS y LLBACO
    plt.figure(figsize=(9, 6))
    # Aquí es donde usamos df_filtered_aco
    df_filtered_aco.groupby('algorithm')['route_load'].mean().plot(
        kind='bar',
        color=['darkgreen', 'indianred']
    )
    plt.title(f'Carga Promedio de la RUTA (LLBACS vs LLBACO) (Snapshot {snapshot_num})')
    plt.xlabel('Algoritmo')
    plt.ylabel('Carga Promedio de la Ruta')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

else:
    print("\nNo se pueden generar gráficos sin ambos conjuntos de datos. Por favor, asegúrate de que los archivos existan y contengan datos válidos.")