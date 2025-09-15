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
"""
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



#----------------------------------------------------eliminar duplicados antes de separar la data--------------
# eliminar duplicados exactos (mantener la primera ocurrencia), segun lo que dijo el profe en el grupo 
df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)

#busqueda de casos atipicos, con la data eliminada y la tecnica vista de rangos intercuartilicos:
def detect_outliers_iqr(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)

# ejemplo para todas las numéricas
outlier_flags = {}
#para cada elemento en df_numeric , busque outliers
for c in df_numeric:
    flags = detect_outliers_iqr(df_clean[c].dropna())
    outlier_flags[c] = flags.sum()
    print(f"{c}: posibles outliers IQR = {flags.sum()}")
"""
 #----------------------------------------------------Division de la data stratified , strada = grupos de cuartiles--------------

 #Como el dataset es de rendimiento academico , una opcion es crear los subgrupos en base a rendimiento academico bajo una medida predefinida
 #por ejemplo 3 grupos, reprobados, rendimiento medio, rendimiento alto, sin embargo estas medidas tienen sesgo ya que dependen de "que es considerado como 
 #rendimiento medio o rendimiento alto , es decir criterio personal"

 # La manera más precisa de division : cuartiles (da mejores resultados en el modelo)

#metodo para generar cuartiles, recomendado 3, al ser la data total de 9873 muestras
def crear_strada(df,target,num_cuartiles):
    #genera los cuartiles en el mismo dataframe df, df es el dataframe SIN REPETIDOS 
    df['strata'] = pd.qcut(df[target], q=num_cuartiles, labels=False, duplicates='drop') #elimina duplicados en empates (dejar los empates o no?)
    # fallback si quedaron NaN (caso raro)
    if df['strata'].isnull().any():
        print("se genero data NaN")
    else:
        df['strata'] = df['strata'].astype(int)
    # reasignar labels contiguos (0..k-1) por si qcut redujo bins
    unique = sorted(df['strata'].unique())
    mapping = {old:new for new, old in enumerate(unique)}
    df['strata'] = df['strata'].map(mapping).astype(int)
    return df

df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)
df_cuartiles =crear_strada(df_clean,'Performance Index',3) #me genera 3 cuartiles con muestras de  3291 cada una
print(df_cuartiles['strata'].value_counts().sort_index()) #imprime las filas strata y su cantidad de elementos
print(df_cuartiles[['Performance Index', 'strata']].head(20)) #imprime los primeros 20 elementos y el grupo percentil asociado
print(df_cuartiles.groupby('strata')['Performance Index'].agg(['count','min','mean','max'])) #imprime minimos, maximos y mediana de cada cuartil 






  