# Инструкция по миграции данных Qdrant и Redis (Snapshots)

Эта инструкция описывает надежный процесс переноса данных Qdrant и Redis с удаленного сервера на локальную машину с использованием механизма снапшотов.

> **Важно:** Все операции выполняются через Docker Compose.

## 1. Подготовка на удаленном сервере (Экспорт)

### Qdrant (через Snapshots)
1. **Создайте снапшот** коллекции:
   ```bash
   curl -X POST 'http://localhost:6333/collections/yandex_handbook_child_chunks/snapshots'
   ```
2. **Найдите имя файла** снапшота:
   ```bash
   curl -s 'http://localhost:6333/collections/yandex_handbook_child_chunks/snapshots' | jq
   ```
3. **Скопируйте файл** из контейнера на хост:
    ```bash
    # Замените <snapshot_name> на имя из предыдущего шага
    docker cp qdrant:/qdrant/snapshots/yandex_handbook_child_chunks/<snapshot_name> ./qdrant_backup.snapshot
    ```
    
    **Примечание:** Snapshots хранятся в `/qdrant/snapshots/` внутри контейнера qdrant, а не в volume `rag_qdrant_storage`.

### Redis (через RDB)
1. **Выполните сохранение** данных:
   ```bash
   docker exec redis redis-cli save
   ```
2. **Скопируйте файл** из контейнера на хост:
   ```bash
   docker cp redis:/data/dump.rdb ./redis_backup.rdb
   ```
3. **Создайте архив** файла `dump.rdb`:
   ```bash
   tar czf redis_backup.tar.gz redis_backup.rdb
   ```

### Проверка экспортированных файлов

Убедитесь, что файлы успешно созданы:
```bash
ls -lh qdrant_backup.snapshot redis_backup.tar.gz
```

Ожидаемый размер:
- `qdrant_backup.snapshot`: ~70 MB
- `redis_backup.tar.gz`: ~2 MB

## 2. Загрузка файлов в Google Drive

### Вариант 1: Через веб-интерфейс

Загрузите файлы `qdrant_backup.snapshot` и `redis_backup.tar.gz` в папку Google Drive:
`https://drive.google.com/drive/folders/1u3HgncinzsNB70NsSdIleWl2ol3jihSg?usp=sharing`

### Вариант 2: Через gdown (если установлен)

```bash
# Установите gdown если нет
pip install gdown

# Загрузите файлы в Google Drive через веб-интерфейс
# или используйте Google Drive API для автоматической загрузки
```

## 3. Скачивание файлов на текущий ПК

Для автоматической загрузки из Google Drive на локальный компьютер:
```bash
docker run --rm -v $(pwd)/rag:/backups ghcr.io/lifelong-learning-assisttant/rag-backup-downloader:v001
```

Файлы будут скачаны в папку `./rag`.

## 3. Подготовка на текущем ПК (Импорт)

### 1. Запуск чистых сервисов
```bash
cd rag
docker-compose up -d qdrant redis
```

### 2. Восстановление Qdrant (Snapshot)
Загрузите файл снапшота в API:
```bash
curl -X POST 'http://localhost:6333/collections/yandex_handbook_child_chunks/snapshots/upload?priority=snapshot' \
    -H 'Content-Type: multipart/form-data' \
    -F 'snapshot=@./rag/qdrant_backup.snapshot'
```

### 3. Восстановление Redis (RDB)
1. **Остановите Redis**:
   ```bash
   docker-compose stop redis
   ```
2. **Распакуйте `dump.rdb`** в volume:
   ```bash
   docker run --rm -v rag_redis_data:/data -v $(pwd)/rag:/backup alpine sh -c "tar xzf /backup/redis_backup.tar.gz -C /data"
   ```
3. **Запустите Redis**:
   ```bash
   docker-compose start redis
   ```

## 4. Запуск сервисов

Теперь можно запустить всю инфраструктуру:

```bash
cd rag
docker-compose up -d
```

## Проверка
После запуска вы можете проверить доступность данных в Qdrant Dashboard (если порты проброшены) или через API:
- HTTP: `http://localhost:6333`
- Коллекции: `http://localhost:6333/collections`

Проверьте количество точек в коллекции:
```bash
curl -s 'http://localhost:6333/collections/yandex_handbook_child_chunks' | jq '.result.points_count'
```

Проверьте данные в Redis:
```bash
docker exec redis redis-cli DBSIZE
```

## Устранение неполадок

### Qdrant: Snapshot не найден

Если при копировании snapshot возникает ошибка "No such file or directory", проверьте фактический путь внутри контейнера:

```bash
docker exec qdrant find /qdrant -name "*.snapshot"
```

Путь должен быть: `/qdrant/snapshots/{collection_name}/{snapshot_name}.snapshot`

### Qdrant: Коллекция уже существует

Если коллекция уже существует на целевом сервере, удалите её перед загрузкой snapshot:

```bash
curl -X DELETE 'http://localhost:6333/collections/yandex_handbook_child_chunks'
```

Затем загрузите snapshot снова.

### Redis: Данные не загружаются

Убедитесь, что Redis остановлен перед копированием файла RDB:

```bash
docker-compose stop redis
docker run --rm -v rag_redis_data:/data -v $(pwd)/rag:/backup alpine sh -c "tar xzf /backup/redis_backup.tar.gz -C /data"
docker-compose start redis
```

Проверьте права доступа к файлу `dump.rdb`:

```bash
docker run --rm -v rag_redis_data:/data alpine ls -la /data/dump.rdb
```

### Qdrant: Ошибка при загрузке snapshot

Если при загрузке snapshot возникает ошибка, убедитесь, что:
1. Qdrant запущен и доступен: `curl http://localhost:6333`
2. Коллекция не существует или удалена
3. Файл snapshot не поврежден

Попробуйте загрузить snapshot с приоритетом `snapshot`:

```bash
curl -X POST 'http://localhost:6333/collections/yandex_handbook_child_chunks/snapshots/upload?priority=snapshot' \
    -H 'Content-Type: multipart/form-data' \
    -F 'snapshot=@./rag/qdrant_backup.snapshot'
```

## Важные примечания

- **Snapshots Qdrant** хранятся в `/qdrant/snapshots/` внутри контейнера qdrant, а не в volume `rag_qdrant_storage`
- Путь к snapshot: `/qdrant/snapshots/{collection_name}/{snapshot_name}.snapshot`
- Всегда проверяйте, что snapshot успешно создан перед копированием
- Snapshots являются согласованными и надежными для операций резервного копирования и восстановления
- Файлы RDB Redis должны копироваться при остановленном Redis для обеспечения согласованности данных
- Размеры файлов: `qdrant_backup.snapshot` ~70 MB, `redis_backup.tar.gz` ~2 MB