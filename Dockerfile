FROM python:3.10-slim

# Dependencias necesarias para compilar extensiones Cython
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar proyecto
COPY . .

# Compilar Cython (si corresponde)
RUN if [ -f v3_cython_parallel_cpu/setup.py ]; then \
        cd v3_cython_parallel_cpu && python setup.py build_ext --inplace; \
    fi

CMD ["/bin/bash"]