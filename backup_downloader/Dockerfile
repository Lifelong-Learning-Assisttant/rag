FROM ghcr.io/astral-sh/uv:0.9.24-python3.13-bookworm-slim

WORKDIR /app

# Устанавливаем curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Копируем файлы проекта
COPY pyproject.toml .
COPY download.py .

# Устанавливаем зависимости
RUN uv sync --no-install-project

# Создаем директорию для бэкапов
RUN mkdir /backups

# Запуск скрипта через uv
CMD ["uv", "run", "python", "download.py"]