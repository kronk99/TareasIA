# U-Net Autoencoder para Reconstruccion de Imagenes

Proyecto de reconstruccion de imagenes usando U-Net Autoencoder con el dataset MVTec AD.
Implementado con PyTorch Lightning, optimizado para Google Colab.

## Modelo Implementado

**U-Net Autoencoder**: Arquitectura U-Net con skip connections para preservar detalles espaciales durante la reconstruccion.

### Caracteristicas del Modelo
- Skip connections entre encoder y decoder
- Espacio latente comprimido (128 dimensiones por defecto)
- Funcion de perdida L1 (MAE) - robusta a outliers
- Normalizacion por lotes opcional
- Optimizador Adam con scheduler ReduceLROnPlateau

## Estructura del Proyecto

```
tarea5/
├── autoencoder_unet_complete.ipynb   # Notebook principal (Google Colab)
├── dataset/                          # Dataset MVTec AD (opcional local)
├── README.md                         # Este archivo
└── .gitignore
```

**Nota**: Todo el codigo esta contenido en el notebook. No hay archivos de codigo fuente separados.

## Dataset

- **MVTec AD** (Anomaly Detection)
- **Clases seleccionadas**: cable, capsule, screw, transistor
- **Tamano de imagen**: 128x128 RGB
- **Total de imagenes**: ~1000 imagenes de entrenamiento

### Preparar el Dataset para Google Colab

1. Visita [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
2. Descarga las clases: **cable, capsule, screw, transistor**
3. Sube los archivos a tu Google Drive en la carpeta `dataset/` con esta estructura:

```
MyDrive/
└── dataset/
    ├── cable/
    │   └── train/
    │       └── good/*.png
    ├── capsule/
    │   └── train/
    │       └── good/*.png
    ├── screw/
    │   └── train/
    │       └── good/*.png
    └── transistor/
        └── train/
            └── good/*.png
```

**Importante**: Solo necesitas las imagenes de `train/good/` para el entrenamiento.

## Uso en Google Colab

### Pasos Rapidos

1. **Subir el notebook a Google Drive**
   - Sube `autoencoder_unet_complete.ipynb` a tu Google Drive

2. **Abrir en Google Colab**
   - Click derecho > Abrir con > Google Colaboratory

3. **Configurar Runtime con GPU**
   - Menu: Runtime > Change runtime type > GPU (T4 recomendado)

4. **Ejecutar las celdas en orden**
   - La primera celda monta Google Drive
   - La segunda celda instala dependencias
   - Continua ejecutando secuencialmente

### Estructura del Notebook

El notebook contiene:
1. Verificacion de GPU y configuracion
2. Montaje de Google Drive
3. Instalacion de dependencias
4. Imports y configuracion
5. Implementacion del dataset MVTec
6. Implementacion del U-Net Autoencoder
7. Funciones de utilidad
8. Carga y visualizacion del dataset
9. Configuracion de W&B (opcional)
10. Entrenamiento del modelo
11. Visualizacion de reconstrucciones
12. Evaluacion cuantitativa (MAE, PSNR, SSIM)
13. Resumen y conclusiones

## Configuracion

### Modificar Parametros

Los parametros estan en la celda 5 (Configuracion YAML). Puedes modificar:

**Modelo:**
```yaml
model:
  latent_dim: 256        # Cambiar de 128 a 256
  learning_rate: 0.0005  # Ajustar learning rate
  unet:
    depth: 5             # Mas profundo (4 por defecto)
    base_channels: 128   # Mas canales (64 por defecto)
```

**Entrenamiento:**
```yaml
trainer:
  max_epochs: 100        # Mas epocas (50 por defecto)
  batch_size: 16         # Reducir si hay problemas de memoria
  num_workers: 2         # Mantener en 2 para Colab
```

**Dataset:**
```yaml
dataset:
  path: /content/drive/MyDrive/dataset  # Ruta a tu dataset
  img_size: 128          # Tamano de imagenes
```

## Caracteristicas

- U-Net Autoencoder con skip connections
- PyTorch Lightning para codigo modular
- Configuracion inline con Hydra/OmegaConf
- W&B para tracking de experimentos (opcional)
- Optimizado para Google Colab (GPU T4/P100/V100)
- Funcion de perdida L1 (MAE)
- Metricas: MAE, PSNR, SSIM
- Callbacks: EarlyStopping, ModelCheckpoint
- Data augmentation (flip, rotation, color jitter)
- Visualizaciones: reconstrucciones, mapas de error

## Resultados Esperados

Despues del entrenamiento (50 epocas):
- **MAE**: ~0.02-0.05 (menor es mejor)
- **PSNR**: ~25-35 dB (mayor es mejor)
- **SSIM**: ~0.85-0.95 (mas cercano a 1.0 es mejor)

El modelo reconstruye imagenes preservando detalles finos gracias a las skip connections.

## Solucion de Problemas

**Error de GPU/CUDA:**
- Verifica: Runtime > Change runtime type > GPU

**Error de memoria (OOM):**
- Reduce `batch_size` a 16 u 8
- Reduce `depth` del U-Net a 3

**Dataset no encontrado:**
- Verifica que la carpeta `dataset` este en MyDrive
- Ejecuta la celda 2 y revisa el output

**W&B no funciona:**
- Ejecuta `!wandb login` en una celda
- O activa `OFFLINE_MODE = True` en la celda 10

## Requisitos

- Cuenta de Google (para Google Colab)
- Dataset MVTec AD en Google Drive
- GPU recomendado (T4 gratis en Colab)
- Cuenta de W&B (opcional, para tracking)

## Licencia

Proyecto academico - TEC
