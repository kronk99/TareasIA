import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mstats
# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('dataset/Student_Performance.csv')
# Explorar estructura
#print(df.info())

# Estadísticas descriptivas (media, desvestandar, min max por columna)
#print(df.describe())
##seccion de histogramas para la exploracion numerica, x representa la variable independiente o caracterisitca, para este caso la y es la frecuencia o conteo
#no el rendimiento academico segun se entiende
df.hist(figsize=(10,8), bins=20)
plt.suptitle('Histogramas de Variables Numéricas')
plt.show()

columnas_a_graficar= ['Hours Studied', 'Previous Scores', 'Sleep Hours' ,'Sample Question Papers Practiced' , 'Performance Index']
fig, axes = plt.subplots(1, len(columnas_a_graficar), figsize=(12, 4))

for i, col in enumerate(columnas_a_graficar):
    plt.yticks(np.arange(0,100,5))
    plt.grid(True)
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()
#para el scatterplot
columnas_a_grarficar2= ['Hours Studied', 'Previous Scores', 'Sleep Hours' ,'Sample Question Papers Practiced' ]
for col in columnas_a_grarficar2:
    df.plot.scatter(x=col, y='Performance Index', alpha=0.6, color='blue')

    plt.title(f'Diagrama de dispersión de {col} y Performance Index')
    plt.xlabel(col)
    plt.ylabel('Performance Index')
    plt.grid(True)
    plt.show()
#matriz de correlacion y grafico de calor
# Calcular matriz de correlación
df_numeric = df.select_dtypes(include=[np.number]) #para datos solo numericos
corr = df_numeric.corr()

fig, ax = plt.subplots(figsize=(10, 8))

# Mostrar matriz como imagen con colores
cax = ax.matshow(corr, cmap='coolwarm')

# Añadir barra de colores a la derecha
fig.colorbar(cax)

# Etiquetas en x e y
ticks = np.arange(len(corr.columns))
ax.set_xticks(ticks)
ax.set_yticks(ticks)

# Rotar etiquetas para que no se encimen
ax.set_xticklabels(corr.columns, rotation=65, ha='left', fontsize=10)
ax.set_yticklabels(corr.columns, fontsize=10)

# Anotar valores en cada celda
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=8)

plt.title('Mapa de calor de la matriz de correlación')
plt.tight_layout()
plt.show()
#busqueda de datos duplicados -------------------------------------------------------
# filas exactamente iguales
n_dup_total = df.duplicated().sum()
print("Duplicados exactos (filas completas):", n_dup_total)

# mostrar algunas duplicadas
dup_rows = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))

print(dup_rows.head(20).to_string())