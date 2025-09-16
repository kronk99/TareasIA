import numpy as np
import pandas as pd

# ---------- Hyperparams ----------
csvPath = "C:/Users/Kevin/Documents/Semestres/II 2025/IA/Tarea 1 Notebook/dataset/Student_Performance.csv"
targetCol = "Performance Index"               
stratumCol = "performanceStratum"             # columna usada para stratified
batchSize = 32
learningRate = 0.01
nEpochs = 20
randomState = 42

df = pd.read_csv(csvPath) # Carga la data



if "performanceStratum" not in df.columns:  # Si no tienes strato creado, crea uno por cuartiles
    df["performanceStratum"] = pd.qcut(df[targetCol], q=4, labels=[f"Q{i+1}" for i in range(4)], duplicates="drop")

# Seleccionar columnas de feature numéricas (excluir target y strato)
numCols = df.select_dtypes(include=[np.number]).columns.tolist()
numCols = [c for c in numCols if c != targetCol]

X_all = df[numCols].to_numpy(dtype=float)
y_all = df[targetCol].to_numpy(dtype=float).reshape(-1, 1)

# Estandarizar features para gradient descent
featureMeans = X_all.mean(axis=0)
featureStds = X_all.std(axis=0, ddof=0)
featureStds[featureStds == 0] = 1.0 # evita division por 0
X_all = (X_all - featureMeans) / featureStds

# Agregar columna intercept
ones = np.ones((X_all.shape[0], 1), dtype=float)
X_all = np.hstack([ones, X_all])

# Generador stratified minibatch
def stratifiedMiniBatchGenerator(dfInput, featuresCols, targetColName, stratumColName,
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

# Training loop: Mini-Batch Gradient Descent #
# Inicializar pesos
p = X_all.shape[1]   # número de columnas después de añadir intercept
w = np.zeros((p, 1), dtype=float)  # vector de pesos iniciales

# Para crear batches, usaremos el DataFrame original 'df'
for epoch in range(1, nEpochs + 1):
    batchGenerator = stratifiedMiniBatchGenerator(df, numCols, targetCol, stratumCol, batchSize=batchSize, randomState=randomState + epoch)
    epochLosses = []
    for Xb, yb in batchGenerator:
        m = Xb.shape[0]
        yPred = Xb.dot(w)            # Predicción
        err = yPred - yb             # Error
        loss = (err ** 2).mean()     # Pérdida (MSE) el batch
        epochLosses.append(loss.item())
        grad = (2.0 / m) * (Xb.T.dot(err))   # Gradient
        w = w - learningRate * grad  # Actualiza pesos

    print(f"Epoch {epoch}/{nEpochs} - avg batch loss: {np.mean(epochLosses):.6f}")

# After training
# Prepara X_full
y_pred_full = X_all.dot(w)
mse_full = np.mean((y_pred_full - y_all) ** 2)
print("Final MSE on full dataset (approx):", mse_full)
