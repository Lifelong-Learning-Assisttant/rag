# LLM Client (OpenAI / OpenRouter / Mistral)

Что это: тонкий клиент поверх LangChain для чата и эмбеддингов с логами и ретраями.

### API:
• validate_api_key() -> (bool, str) — живая проверка ключа
• generate(texts, model=None, api_key=None, **kw) -> List[str] — промпты → ответы
• embed(texts, model=None, api_key=None, **kw) -> List[List[float]] — тексты → векторы
• create_chat() / create_embeddings() — фабрики клиентов

### Установка (через uv):

```bash
# 1) создать и активировать venv
uv venv
# 2) установить зависимости
uv sync
# 3) запустить мини-тесты
uv run settings.py
uv run llm_client.py
```

### Пример:

```python
from settings import get_settings
from llm_client import LLMClient

cfg = get_settings()
client = LLMClient(provider=cfg.default_provider)  # "openai" | "openrouter" | "mistral"
print(client.validate_api_key())
print(client.generate(["ping"], temperature=0.0))
print([len(v) for v in client.embed(["hello", "world"])])
```