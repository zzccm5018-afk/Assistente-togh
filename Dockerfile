# Dockerfile para hospedar main.py (+ admin_panel.py) no Hugging Face Spaces
FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema (se precisar compilar algo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro (cache de camadas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código
COPY . .

# Hugging Face Spaces espera a aplicação na porta 7860
EXPOSE 7860

# Sobe o main.py (que já inclui o admin_panel.py se você adicionar o include_router)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
