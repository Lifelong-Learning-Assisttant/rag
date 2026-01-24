#!/bin/sh
set -e

echo '📥 Downloading backups...'
uv run python download.py

echo '⏳ Waiting for Qdrant...'
# Используем переменную окружения для хоста Qdrant или дефолтное значение
QDRANT_HOST="${QDRANT_HOST:-rag-qdrant}"
QDRANT_PORT="${QDRANT_PORT:-6333}"

for i in $(seq 1 30); do
  if curl -s "http://${QDRANT_HOST}:${QDRANT_PORT}/" | grep -q 'qdrant'; then
    echo '✅ Qdrant is ready!'
    break
  fi
  echo "📡 Qdrant not ready yet (attempt $i/30)..."
  sleep 5
  if [ $i -eq 30 ]; then echo '❌ Qdrant timeout!'; exit 1; fi
done

echo '🚀 Importing Qdrant Snapshot...'
curl -X POST "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/yandex_handbook_child_chunks/snapshots/upload?priority=snapshot" \
     -H 'Content-Type: multipart/form-data' \
     -F 'snapshot=@/backups/qdrant_backup.snapshot'

echo '📦 Extracting Redis RDB...'
# Удаляем старый AOF, чтобы Redis гарантированно грузился с RDB
if [ -f /redis_data/appendonly.aof ]; then
    echo "🗑️ Removing old appendonly.aof..."
    rm -f /redis_data/appendonly.aof
fi

tar xzf /backups/redis_backup.tar.gz -C /redis_data

# Переименовываем восстановленный файл в dump.rdb, если он называется иначе
if [ -f /redis_data/redis_backup.rdb ]; then
    echo "🔄 Renaming redis_backup.rdb to dump.rdb..."
    mv /redis_data/redis_backup.rdb /redis_data/dump.rdb
fi

# Исправляем права доступа (Redis в контейнере работает под uid 999)
echo '🔧 Fixing permissions...'
chown -R 999:999 /redis_data

if [ -f /redis_data/dump.rdb ]; then
  echo "✅ dump.rdb found in /redis_data (size: $(du -h /redis_data/dump.rdb | cut -f1))"
else
  echo "⚠️ WARNING: dump.rdb NOT found in /redis_data after extraction!"
  ls -la /redis_data
fi

echo '✅ Bootstrap completed successfully!'