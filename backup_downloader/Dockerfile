FROM python:3.13-slim-bookworm

# Устанавливаем curl и uv
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv==0.9.26

WORKDIR /app

# Копируем файлы проекта
COPY pyproject.toml .
COPY download.py .

# Устанавливаем зависимости
RUN uv sync --no-install-project

# Создаем директорию для бэкапов
RUN mkdir /backups

# Запуск скрипта через uv
CMD ["uv", "run", "python", "download.py"]