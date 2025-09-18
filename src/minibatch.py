import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ---------------- Config ----------------
csvPath = "dataset/Student_Performance.csv"
target_col = "Performance Index"
features = [
    'Hours Studied',
    'Previous Scores',
    'Sample Question Papers Practiced',
    'Sleep Hours',
    'Extracurricular Activities'
]
random_state = 42
train_frac, val_frac, test_frac = 0.70, 0.15, 0.15
batch_size = 64
learning_rate = 1e-3 #esto es 0.001
n_epochs = 50

# ---------------- util: crear strata (terciles) ----------------
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
def separacion_estratificada(df, strata_col='strata', train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8
    train_parts, val_parts, test_parts = [], [], []
    for g, group in df.groupby(strata_col, observed=True):
        group_shuffled = group.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(group_shuffled)
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))
        # rest -> test
        n_test = n - n_train - n_val
        train_parts.append(group_shuffled.iloc[:n_train])
        val_parts.append(group_shuffled.iloc[n_train:n_train+n_val])
        test_parts.append(group_shuffled.iloc[n_train+n_val:])
    df_train = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_val = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_test = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_train, df_val, df_test
#normalizacion de los datos.
def prepare_X_from_stats(df_part, features, featureMeans, featureStds):
    X = df_part[features].to_numpy(dtype=float)
    X_norm = (X - featureMeans) / featureStds
    ones = np.ones((X_norm.shape[0], 1), dtype=float)
    return np.hstack([ones, X_norm])  # intercept + normalized features
def stratifiedMiniBatchGenerator(dfInput, featuresCols, targetColName, stratumColName,
                                 featureMeans, featureStds,
                                 batchSize=32, randomState=42):
    """
    Generador que en cada llamada produce (x_batch, y_batch).
    Mantiene la proporción de estratos en cada batch aproximadamente.
    """
    rng = np.random.RandomState(randomState)
    dfShuffled = dfInput.sample(frac=1, random_state=randomState).reset_index(drop=True)

    groupSizes = dfShuffled.groupby(stratumColName, observed=True).size()
    proportions = (groupSizes / len(dfShuffled)).to_dict()  # proporciones por strato

    # convierte el DataFrame en una lista de índices por strato para muestras rápidas
    idxByStratum = {g: dfShuffled[dfShuffled[stratumColName] == g].index.tolist()
                    for g in dfShuffled[stratumColName].unique()}

    pointerByStratum = {g: 0 for g in idxByStratum} # pointer index por strato

    totalRemaining = len(dfShuffled)

    while totalRemaining >= batchSize:
        chosenIndices = []
        for stratum, prop in proportions.items():   # para cada strato, toma la cantidad esperada en el batch
            nExpected = int(round(prop * batchSize))
            availableIndices = idxByStratum[stratum]
            ptr = pointerByStratum[stratum]
            nAvailable = len(availableIndices) - ptr
            nTake = min(nExpected, nAvailable)
            if nTake > 0:
                chosen = availableIndices[ptr: ptr + nTake]
                chosenIndices.extend(chosen)
                pointerByStratum[stratum] += nTake

        # Si por redondeos no alcanza el tamaño del batch, rellena con índices aleatorios de los que quedan
        if len(chosenIndices) < batchSize:
            leftover = []
            for stratum, inds in idxByStratum.items():  # construye lista de todos índices no usados aún
                ptr = pointerByStratum[stratum]
                leftover.extend(inds[ptr:])
            if len(leftover) >= (batchSize - len(chosenIndices)):
                extra = rng.choice(leftover, size=(batchSize - len(chosenIndices)), replace=False).tolist()
                chosenIndices.extend(extra)
            else:   # sale si no hay suficientes leftovers
                break

        
        chosenSet = set(chosenIndices)  # Convierte chosenIndices a conjunto para facilitar
        for stratum in list(idxByStratum.keys()):   # Reconstruye el id por strato eliminando items de chosenSet
            lst = idxByStratum[stratum]
            newLst = [i for i in lst if i not in chosenSet]
            idxByStratum[stratum] = newLst
            pointerByStratum[stratum] = 0  # resetea el pointer

        totalRemaining = sum(len(v) for v in idxByStratum.values())

        # obtiene X_batch, y_batch
        batchDf = dfShuffled.loc[chosenIndices]
        Xb = batchDf[featuresCols].to_numpy(dtype=float)
        yb = batchDf[targetColName].to_numpy(dtype=float).reshape(-1, 1)

        # estandardizamos con los mismos stats y agregamos intercept
        Xb = (Xb - featureMeans) / featureStds
        ones_b = np.ones((Xb.shape[0], 1), dtype=float)
        Xb = np.hstack([ones_b, Xb])

        yield Xb, yb




df = pd.read_csv('dataset/Student_Performance.csv')
df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)
# 2) convertir categóricas relevantes ANTES del split (p.ej. Yes/No -> 1/0)
if 'Extracurricular Activities' in df_clean.columns:
    df_clean['Extracurricular Activities'] = df_clean['Extracurricular Activities'].map({'Yes':1,'No':0})
df_cuartiles =crear_strada(df_clean,'Performance Index',3)
df_train, df_val, df_test = separacion_estratificada(df_cuartiles, strata_col='strata',train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,random_state=random_state)
#quitar las columnas llamadas strada
df_train_with_strata = df_train.copy() #se necesita una copia para el for de entrenamiento,con la columna con strada
df_train = df_train.drop(columns=['strata']).reset_index(drop=True)#para el calculo de freaturemeans y freaturestds
df_val   = df_val.drop(columns=['strata']).reset_index(drop=True)
df_test  = df_test.drop(columns=['strata']).reset_index(drop=True)
# 5) calcular featureMeans/featureStds SOLO sobre df_train
X_train_raw = df_train[features].to_numpy(dtype=float)
featureMeans = X_train_raw.mean(axis=0)
featureStds  = X_train_raw.std(axis=0, ddof=0)
featureStds[featureStds == 0] = 1.0
# 6) preparar X_val, X_test ((son los datos matriciales validados)

X_val = prepare_X_from_stats(df_val, features, featureMeans, featureStds)
y_val = df_val[target_col].to_numpy(dtype=float).reshape(-1,1)
X_test = prepare_X_from_stats(df_test, features, featureMeans, featureStds)
y_test = df_test[target_col].to_numpy(dtype=float).reshape(-1,1)
# 7) inicializar pesos (p = n_features + 1)
p = 1 + len(features)
w = np.zeros((p,1), dtype=float)
train_grafica = []
val_grafica = []
# 8) entrenamiento usando el generador en el trainining
for epoch in range(1, n_epochs + 1):
    
    gen = stratifiedMiniBatchGenerator(df_train_with_strata, features, target_col, 'strata',featureMeans, featureStds,batchSize=batch_size, randomState=random_state)
    epoch_losses = []
    for Xb, yb in gen: #calculo de pesos por bache
        m_b = Xb.shape[0]
        yPred = Xb.dot(w)
        err = yPred - yb
        loss = np.mean(err**2)
        epoch_losses.append(loss)
        grad = (2.0 / m_b) * (Xb.T.dot(err))
        w = w - learning_rate * grad

    # calcular val loss (usando X_val preparado con stats de train)
    train_loss = np.mean(epoch_losses)
    val_pred = X_val.dot(w)
    val_loss = np.mean((val_pred - y_val)**2)
    #guardado de datos por epoch
    train_grafica.append(train_loss)
    val_grafica.append(val_loss)

    print(f"Epoch {epoch}/{n_epochs} - MSE_PROMEDIO_TESTIG={np.mean(epoch_losses):.6f}, MSE_PROMEDIO_VALIDACION={val_loss:.6f}")
# ----- Graficas de training vs validation de mini batch gradient descent -----
plt.figure(figsize=(8,5))
plt.plot(range(1, n_epochs+1), train_grafica, label="Training MSE")
plt.plot(range(1, n_epochs+1), val_grafica, label="Validation MSE")
plt.xlabel("Épocas")
plt.ylabel("MSE")
plt.title("Curvas de aprendizaje: Training vs Validation")
plt.legend()
plt.grid(True)
plt.show()