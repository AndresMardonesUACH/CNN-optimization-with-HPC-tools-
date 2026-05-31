# CNN-HPC

Proyecto de optimización de una Red Neuronal Convolucional (CNN) utilizando técnicas de High Performance Computing (HPC).  
El objetivo es implementar y comparar distintas versiones de entrenamiento e inferencia de una CNN, evaluando rendimiento, paralelización y aceleración.

---

## Estructura del proyecto

```text
cnn-hpc/
│
├── data/
│   ├── raw/              # Dataset original
│   ├── processed/        # Datos preprocesados
│   └── downloads/        # Descargas automáticas del dataset
│
├── src/
│   ├── train.py          # Entrenamiento principal
│   ├── model.py          # Arquitectura de la CNN
│   └── dataset.py        # Carga y preprocesamiento del dataset
│
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

---

## Objetivos del proyecto

- Implementar una CNN base para clasificación de imágenes.
- Comparar distintas estrategias de optimización.
- Medir tiempos de entrenamiento e inferencia.
- Aplicar técnicas HPC como:
  - Paralelización en CPU
  - Uso de GPU
  - Vectorización
  - JIT compilation
  - Batch processing

---

## Dataset

El proyecto utilizará un dataset de clasificación de imágenes.  
Inicialmente se trabajará con:

- CIFAR-10

Los datasets se almacenan en:

```text
data/raw/
```

---

## Requisitos

- Python 3.11 o superior
- pip
- Entorno virtual (`venv`)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/AndresMardonesUACH/CNN-optimization-with-HPC-tools-.git
cd cnn-hpc
```

### 2. Crear entorno virtual

Linux / WSL:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución

Entrenar la CNN:

```bash
python src/train.py
```

---

## Tecnologías consideradas

Dependiendo de la etapa del proyecto, se evaluará el uso de:

- NumPy
- PyTorch
- JAX
- Cython

---

## Métricas a evaluar

- Accuracy
- Loss
- Tiempo de entrenamiento
- Tiempo de inferencia
- Uso de CPU/GPU
- Escalabilidad

---

## Posibles etapas de optimización

1. Implementación base en Python
2. Vectorización con NumPy
3. Optimización con Cython
4. Paralelización en CPU
5. Aceleración con GPU
6. Comparación de rendimiento

---

## Autor

Andrés Mardones Domcke, estudiante de Ingeniería Civil en Informática.