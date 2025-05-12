# Base Python
FROM python:3.10-slim

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . /app

# Instala as dependências do Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Porta da API
EXPOSE 8000 5000

# Comando padrão para iniciar a API
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & mlflow server --host=0.0.0.0 --port=5000"]