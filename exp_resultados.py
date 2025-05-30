import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # Para listar archivos y unir rutas

# --- Configuración ---
# Directorio donde se encuentran tus archivos CSV
csv_directory = './configuraciones'
# --- Fin modificación ---

# Prefijo y sufijo de tus archivos CSV
csv_filename_prefix = 'conf_'
csv_filename_suffix = '.csv'

# Número de configuraciones (archivos CSV)
# Asegúrate de que este número coincide con la cantidad de archivos conf_X.csv que tienes (ej. conf_1.csv a conf_9.csv)
num_configurations = 9 

# --- Análisis ---

# Lista para almacenar los datos resumen de cada configuración
summary_data = []

# Verificar si el directorio existe
if not os.path.isdir(csv_directory):
    print(f"Error: El directorio '{csv_directory}' no fue encontrado.")
    print("Asegúrate de que tus archivos CSV están en este subdirectorio.")
else:
    print(f"Buscando archivos CSV en: {os.path.abspath(csv_directory)}")

    # Iterar sobre el número de configuraciones
    for i in range(1, num_configurations + 1):
        filename = f"{csv_filename_prefix}{i}{csv_filename_suffix}"
        filepath = os.path.join(csv_directory, filename) # Usamos os.path.join para construir la ruta correctamente

        if os.path.exists(filepath):
            print(f"Procesando archivo: {filename}")
            try:
                # Leer el archivo CSV
                df = pd.read_csv(filepath)

                # Verificar si el DataFrame no está vacío y contiene las columnas esperadas
                if not df.empty and 'mean' in df.columns and 'std' in df.columns and 'time' in df.columns:
                    # Las columnas 'mean', 'std' y 'time' deberían ser las mismas para todas las filas
                    # dentro de un mismo archivo CSV (por snapshot). Tomamos el valor de la primera fila.
                    mean_cost = df['mean'].iloc[0] 
                    std_cost = df['std'].iloc[0]   
                    mean_time = df['time'].iloc[0] # <-- Nuevo: Extraemos el tiempo

                    # Añadir los datos a la lista resumen
                    summary_data.append({
                        'Configuration': f'Config {i}', # Nombre descriptivo para la configuración
                        'Mean Cost': mean_cost,      
                        'Std Dev Cost': std_cost,    
                        'Mean Time': mean_time       # <-- Nuevo: Añadimos el tiempo al resumen
                    })
                else:
                    print(f"Advertencia: El archivo {filename} está vacío o no contiene las columnas 'mean', 'std' o 'time'.")

            except Exception as e:
                print(f"Error al leer o procesar el archivo {filename}: {e}")
        else:
            print(f"Advertencia: Archivo no encontrado: {filename}")


    # Crear un DataFrame resumen a partir de los datos recopilados
    summary_df = pd.DataFrame(summary_data)

    # Verificar si se recopilaron datos antes de intentar mostrar tablas/gráficos
    if not summary_df.empty:
        # Ordenar por costo medio para una mejor visualización (opcional)
        summary_df = summary_df.sort_values('Mean Cost').reset_index(drop=True)

        # --- Mostrar Tabla de Resultados ---
        print("\n--- Tabla Resumen de Resultados por Configuración ---")
        try:
            print(summary_df.to_markdown(index=False)) 
        except ImportError:
            print("Por favor, instala la biblioteca 'tabulate' (pip install tabulate) para ver la tabla en formato Markdown.")
            print(summary_df.to_string(index=False)) 


        # --- Generar Gráficos ---

        # Configurar el estilo de los gráficos (opcional, mejora la apariencia)
        sns.set_theme(style="whitegrid")

        # Gráfico de Barras para el Costo Medio
        plt.figure(figsize=(10, 6))
        barplot_mean = sns.barplot(x='Configuration', y='Mean Cost', data=summary_df, palette='viridis')
        plt.title('Costo Medio del Camino por Configuración de LLBACO')
        plt.xlabel('Configuración')
        plt.ylabel('Costo Medio del Camino')
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout() 

        for container in barplot_mean.containers:
            plt.bar_label(container, fmt='%.4f') 
        plt.show() # Mostrar este gráfico antes de pasar al siguiente


        # Gráfico de Barras para la Desviación Típica del Costo
        plt.figure(figsize=(10, 6))
        barplot_std = sns.barplot(x='Configuration', y='Std Dev Cost', data=summary_df, palette='viridis')
        plt.title('Desviación Típica del Costo por Configuración de LLBACO')
        plt.xlabel('Configuración')
        plt.ylabel('Desviación Típica del Costo')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for container in barplot_std.containers:
             plt.bar_label(container, fmt='%.4f')
        plt.show() # Mostrar este gráfico


        # --- NUEVO GRÁFICO: Gráfico de Barras para el Tiempo Medio de Ejecución ---
        plt.figure(figsize=(10, 6))
        barplot_time = sns.barplot(x='Configuration', y='Mean Time', data=summary_df, palette='magma')
        plt.title('Tiempo Medio de Ejecución por Configuración de LLBACO')
        plt.xlabel('Configuración')
        plt.ylabel('Tiempo Medio (segundos)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for container in barplot_time.containers:
             plt.bar_label(container, fmt='%.4f') # Muestra el tiempo con 4 decimales
        plt.show() # Mostrar este nuevo gráfico

        print("\nAnálisis completo y gráficos generados.")
    else:
        print("\nNo se pudieron recopilar datos de ningún archivo CSV para generar gráficos.")