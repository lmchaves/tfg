import pandas as pd
import matplotlib.pyplot as plt

# 1. Carga el CSV generado
df = pd.read_csv('aco_experiment_snapshot4.csv')

# 2. Muestra los primeros registros y el resumen estadístico
print(df.head())
print(df[['mean','std']].drop_duplicates())

# 3. Grafica el coste de cada ejecución
plt.figure()
plt.plot(df['run'], df['cost'])
plt.xlabel('Ejecución (run)')
plt.ylabel('Coste')
plt.title('Coste por ejecución (Snapshot 4)')
plt.grid(True)
plt.tight_layout()
plt.show()
