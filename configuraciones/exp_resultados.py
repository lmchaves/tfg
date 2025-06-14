import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

csv_directory = './configuraciones'
csv_filename_prefix = 'conf_'
csv_filename_suffix = '.csv'
num_configurations = 14

summary_data = []

if not os.path.isdir(csv_directory):
    print(f"Error: El directorio '{csv_directory}' no fue encontrado.")
    print("Asegúrate de que tus archivos CSV están en este subdirectorio.")
else:
    print(f"Buscando archivos CSV en: {os.path.abspath(csv_directory)}")

    for i in range(1, num_configurations + 1):
        filename = f"{csv_filename_prefix}{i}{csv_filename_suffix}"
        filepath = os.path.join(csv_directory, filename)

        if os.path.exists(filepath):
            print(f"Procesando archivo: {filename}")
            try:
                df = pd.read_csv(filepath)

                if not df.empty and \
                   'mean_cost' in df.columns and \
                   'std_cost' in df.columns and \
                   'total_aco_time_s' in df.columns:
                     
                    mean_cost = df['mean_cost'].iloc[0] 
                    std_cost = df['std_cost'].iloc[0]    
                    mean_time = df['total_aco_time_s'].iloc[0] 

                    summary_data.append({
                        'Configuration': f'Config {i}',
                        'Mean Cost': mean_cost,     
                        'Std Dev Cost': std_cost,   
                        'Mean Time': mean_time      
                    })
                else:
                    print(f"Advertencia: El archivo {filename} está vacío o no contiene las columnas 'mean_cost', 'std_cost' o 'total_aco_time_s'.")

            except Exception as e:
                print(f"Error al leer o procesar el archivo {filename}: {e}")
        else:
            print(f"Advertencia: Archivo no encontrado: {filename}")

summary_df = pd.DataFrame(summary_data)

if not summary_df.empty:
    summary_df_sorted_by_cost = summary_df.sort_values('Mean Cost').reset_index(drop=True)

    print("\n--- Tabla Resumen de Resultados por Configuración ---")
    try:
        print(summary_df_sorted_by_cost.to_markdown(index=False)) 
    except ImportError:
        print("Por favor, instala la biblioteca 'tabulate' (pip install tabulate) para ver la tabla en formato Markdown.")
        print(summary_df_sorted_by_cost.to_string(index=False)) 

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    ax_mean_cost = summary_df_sorted_by_cost.set_index('Configuration')['Mean Cost'].plot(
        kind='bar',
        yerr=summary_df_sorted_by_cost.set_index('Configuration')['Std Dev Cost'],
        capsize=5,
        color=sns.color_palette('viridis', n_colors=len(summary_df_sorted_by_cost)),
        edgecolor='black' 
    )
    plt.title('Coste Medio del Camino por Configuración de LLBACO')
    plt.xlabel('Configuración')
    plt.ylabel('Coste Medio del Camino')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    for container in ax_mean_cost.containers:
        if hasattr(container, 'patches'):
            plt.bar_label(container, fmt='%.4f')
    plt.show()

    plt.figure(figsize=(10, 6))
    summary_df_sorted_by_std = summary_df.sort_values('Std Dev Cost').reset_index(drop=True)
    barplot_std = sns.barplot(
        x='Configuration', 
        y='Std Dev Cost', 
        data=summary_df_sorted_by_std, 
        palette='viridis'
    )
    plt.title('Desviación Típica del Coste por Configuración de LLBACO')
    plt.xlabel('Configuración')
    plt.ylabel('Desviación Típica del Coste')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    for container in barplot_std.containers:
        plt.bar_label(container, fmt='%.4f')
    plt.show()

    plt.figure(figsize=(10, 6))
    summary_df_sorted_by_time = summary_df.sort_values('Mean Time').reset_index(drop=True)
    summary_df_sorted_by_time['Mean Time'] = summary_df_sorted_by_time['Mean Time'] / 30.0

    barplot_time = sns.barplot(
        x='Configuration', 
        y='Mean Time', 
        data=summary_df_sorted_by_time, 
        palette='magma'
    )
    plt.title('Tiempo Medio de Ejecución por Configuración de LLBACO')
    plt.xlabel('Configuración')
    plt.ylabel('Tiempo Medio (segundos)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    for container in barplot_time.containers:
        plt.bar_label(container, fmt='%.4f')
    plt.show()

    print("\nAnálisis completo y gráficos generados.")
else:
    print("\nNo se pudieron recopilar datos de ningún archivo CSV para generar gráficos.")