# CNN Optimization with HPC Tools

Proyecto académico orientado a la optimización de una Red Neuronal Convolucional (CNN) para clasificación de imágenes mediante técnicas de High Performance Computing (HPC).

El objetivo fue implementar y comparar distintas estrategias de optimización aplicadas al entrenamiento de una CNN, evaluando rendimiento computacional, paralelización y aceleración sobre CPU y GPU.

---

## Descripción

Se desarrolló una CNN simple para clasificación sobre el dataset CIFAR-10 y se implementaron cuatro versiones equivalentes:

1. **Python base**
   Implementación utilizando bucles explícitos en Python como línea base de comparación.

2. **Vectorización con NumPy**
   Reemplazo de operaciones iterativas por operaciones vectorizadas (`tensordot`, `sliding_window_view`, `outer`, etc.).

3. **Paralelización en CPU con Cython + OpenMP**
   Optimización de la capa de convolución mediante compilación y paralelización multinúcleo.

4. **GPU con JAX**
   Ejecución acelerada mediante JIT compilation y procesamiento paralelo en GPU.

5. **Distribuida con RAY**
   Utilizando una tecnología de coordinación con diferentes workers, cada uno encargado de las convoluciones de cada filtro.

El estudio compara el rendimiento de cada versión y analiza el impacto de distintas técnicas HPC aplicadas a redes convolucionales.

---

## Estructura del proyecto

```text
CNN-optimization-with-HPC-tools/
│
├── data/
│   └── CIFAR-10
│
├── v1_baseline_python/
│   ├── layers/
│   ├── model.py
│   ├── train.py
│   └── benchmark.py
│
├── v2_numpy_vectorized/
│   ├── layers/
│   ├── model.py
│   ├── train.py
│   └── benchmark.py
│
├── v3_cython_parallel_cpu/
│   ├── layers/
│   ├── model.py
│   ├── train.py
│   ├── benchmark.py
│   └── setup.py
│
├── v4_jax_parallel_gpu/
│   ├── layers/
│   ├── model.py
│   ├── train.py
│   └── benchmark.py
|
├── v5_ray_parallel_distributed/
│   ├── layers/
│   ├── model.py
│   ├── train.py
│   └── benchmark.py
│
├── requirements.txt
├── benchmark_cnn_hpc_metricas.xlsx
├── informe.pdf
├── Dockerfile
└── README.md
```

---

## Dataset

Se utilizó:

* CIFAR-10

Imágenes:

```text
32 x 32 x 3
```

Clases:

```text
10
```

Fuente:

[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com)

---

## Arquitectura de la CNN

Arquitectura utilizada en las cuatro versiones:

```text
Input (32x32x3)
→ ConvLayer (8 filtros, 3x3)
→ ReLU
→ MaxPool (2x2)
→ Dense
→ Softmax
```

Salida:

```text
10 clases
```

---

## Requisitos

* Python 3.11+
* pip
* entorno virtual
* compilador C/C++ (para Cython)
* OpenMP
* CUDA + GPU NVIDIA (opcional, para JAX)

---

## Instalación

Clonar repositorio:

```bash
git clone https://github.com/AndresMardonesUACH/CNN-optimization-with-HPC-tools.git
cd CNN-optimization-with-HPC-tools
```

Crear entorno virtual:

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

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Compilar Cython

Para la versión CPU paralela:

```bash
cd v3_cython_parallel_cpu
python setup.py build_ext --inplace
```

---

# Docker

También es posible ejecutar el proyecto mediante Docker, evitando instalar manualmente las dependencias.

## Opción 1: Construir la imagen localmente

Desde la raíz del proyecto:

```bash
docker build -t cnn-hpc:1.0 .
```

Verificar la imagen:

```bash
docker images
```

Ejecutar un contenedor interactivo:

```bash
docker run --rm -it cnn-hpc:1.0
```

---

## Opción 2: Descargar la imagen desde Docker Hub

Descargar la imagen:

```bash
docker pull andresmardones/cnn-hpc:1.0
```

Ejecutar:

```bash
docker run --rm -it andresmardones/cnn-hpc:1.0
```

---

## Ejecutar benchmarks

Para ejecutar se necesitan 3 argumentos: Modo, Etapa y número de imágenes
   - Modo: `tiempo` `memoria` `energia`
   - Etapa: `conv` `forward` `train`

Python base:

```bash
python v1_baseline_python/benchmark.py [modo] [etapa] [N]
```

NumPy:

```bash
python v2_numpy_vectorized/benchmark.py [modo] [etapa] [N]
```

Cython CPU:

```bash
python v3_cython_parallel_cpu/benchmark.py [modo] [etapa] [N]
```

JAX GPU:

```bash
python v4_jax_gpu/benchmark.py [modo] [etapa] [N]
```

RAY:

```bash
python v5_ray_parallel_distributed/benchmark.py [modo] [etapa] [N]
```

---

## Métricas evaluadas

Se midieron:

* tiempo de convolución
* forward pass completo
* entrenamiento completo
* energía
* memoria

---

## Resultados generales

Resumen:

* Python base presentó el mayor tiempo de ejecución.
* NumPy obtuvo la mayor aceleración en operaciones vectorizables.
* Cython mejoró significativamente el rendimiento en CPU multinúcleo.
* JAX mostró mejor aprovechamiento del paralelismo sobre GPU en entrenamiento completo.
* RAY presenta claros problemas de sincronización frente a lo simple de la convolución. Aun así mejora la versión base.

Conclusión principal:

> La vectorización redujo drásticamente el costo computacional y el uso de técnicas HPC permitió acelerar el entrenamiento de la CNN en varios órdenes de magnitud respecto a la implementación base.

---

## Tecnologías utilizadas

* Python
* NumPy
* Cython
* JAX
* OpenMP
* Ray
* CIFAR-10

---

## Autor

Andrés Mardones Domcke
Ingeniería Civil en Informática
