# Dockerfile для RAG API сервиса
FROM python:3.13-slim

WORKDIR /app

# Установка uv
RUN pip install uv==0.9.26

# Копирование файлов проекта
COPY pyproject.toml uv.lock* ./

# Установка зависимостей через uv с использованием системного Python
ENV UV_SYSTEM_PYTHON=1
RUN uv pip install --system -e .

# Копирование кода приложения
COPY app/ ./app/
COPY llm_service/ ./llm_service/
COPY logger.py settings.py ./

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
