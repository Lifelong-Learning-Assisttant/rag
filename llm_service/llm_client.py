"""
Модуль подключения к апи моделей
"""

import time
from typing import Any, List, Optional, Sequence, Tuple

from httpx import ConnectError, HTTPStatusError, TimeoutException
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from logger import get_logger
from settings import get_settings
from llm_service.utils import (
    build_httpx_timeout,
    extract_request_id_from_exc,
    openrouter_headers,
    truncate,
    unwrap_http_exc,
)


class LLMClient:
    """
    Клиент для LLM и эмбеддингов (OpenAI / OpenRouter / Mistral) поверх LangChain.
    """

    def __init__(self, provider: str):
        """
        Args:
            provider: Имя провайдера: "openai" | "openrouter" | "mistral".
        """
        self.provider = (provider or "").lower().strip()
        self.cfg = get_settings()
        self.log = get_logger(__name__)
        self.log.info("Инициализация LLM-клиента: провайдер=%s", self.provider)

    # ------------------------- ключ -------------------------

    def _resolve_api_key(self, method_api_key: Optional[str] = None) -> Optional[str]:
        """
        Определяет ключ API: приоритет — аргумент метода, затем settings.

        Args:
            method_api_key: Ключ, переданный в метод.

        Returns:
            Строка ключа или None, если ключ не найден.
        """
        self.log.debug("start:_resolve_api_key провайдер=%s", self.provider)
        if method_api_key:
            self.log.debug("Ключ API получен из аргумента метода")
            return method_api_key

        p = self.provider
        if p == "openai" and self.cfg.openai_api_key:
            return self.cfg.openai_api_key.get_secret_value()
        if p == "openrouter" and self.cfg.openrouter_api_key:
            return self.cfg.openrouter_api_key.get_secret_value()
        if p == "mistral" and self.cfg.mistral_api_key:
            return self.cfg.mistral_api_key.get_secret_value()

        self.log.warning("Ключ API отсутствует для провайдера=%s", self.provider)
        return None

    # -------------------- универсальный ретрай --------------------

    def _is_retriable_exc(self, exc: Exception) -> Tuple[bool, Optional[int]]:
        """
        Решает, стоит ли ретраить исключение.

        Args:
            exc: Исключение из вызова провайдера.

        Returns:
            (retriable, http_status) — признак ретрая и HTTP-статус (если известен).
        """
        if isinstance(exc, (TimeoutException, ConnectError)):
            return True, None

        if isinstance(exc, HTTPStatusError) and exc.response is not None:
            status = exc.response.status_code
            if status == 429 or 500 <= status < 600:
                return True, status

        _, status, _, _, _ = unwrap_http_exc(exc)
        if status is not None and (status == 429 or 500 <= status < 600):
            return True, status

        return False, None

    def _call_with_retry(self, op_name: str, fn: callable) -> Any:
        """
        Выполняет вызов с фиксированными ретраями.

        Args:
            op_name: Имя операции для логов.
            fn: Нулераговая функция, которую нужно выполнить (без аргументов).

        Returns:
            Результат вызова fn().

        Raises:
            Exception: Последняя ошибка, если все попытки исчерпаны.
        """
        self.log.info("start:%s", op_name)
        last_exc: Optional[Exception] = None
        max_tries = 5
        sleep_seconds = 3.0

        for attempt in range(1, max_tries + 1):
            t0 = time.perf_counter()
            self.log.debug("%s: попытка %d/%d", op_name, attempt, max_tries)
            try:
                result = fn()
                dt = (time.perf_counter() - t0) * 1000
                # usage (если модель возвращает метаданные)
                usage = None
                try:
                    rm = getattr(result, "response_metadata", None)
                    if isinstance(rm, dict):
                        usage = rm.get("token_usage") or rm.get("usage")
                except Exception:
                    pass

                if usage:
                    self.log.debug("%s: ok за %.1f мс, usage=%s", op_name, dt, usage)
                else:
                    self.log.debug("%s: ok за %.1f мс", op_name, dt)
                return result

            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000
                exc, status, retry_after, req_id, body = unwrap_http_exc(e)
                retriable, _ = self._is_retriable_exc(exc)

                self.log.warning(
                    "%s: ошибка на попытке %d: %.1f мс, status=%s, retriable=%s, "
                    "retry_after=%s, request_id=%s, body=%s, exc=%s",
                    op_name,
                    attempt,
                    dt,
                    status,
                    retriable,
                    retry_after,
                    req_id or extract_request_id_from_exc(exc),
                    truncate(body),
                    repr(exc),
                )
                last_exc = exc

                if not retriable or attempt == max_tries:
                    break

                self.log.info("%s: повтор через %.0f с", op_name, sleep_seconds)
                time.sleep(sleep_seconds)

        raise last_exc if last_exc else RuntimeError(f"{op_name} failed")

    # ------------------------- фабрики -------------------------

    def _chat_model_for_provider(self, provider: str, override: Optional[str]) -> str:
        """Возвращает имя чат-модели провайдера с учётом override."""
        if override:
            return override
        if provider == "openai":
            return self.cfg.openai_chat_model_name
        if provider == "openrouter":
            return self.cfg.openrouter_chat_model
        if provider == "mistral":
            return self.cfg.mistral_chat_model
        raise ValueError(f"Неподдерживаемый провайдер: {provider}")

    def _emb_model_for_provider(self, provider: str, override: Optional[str]) -> str:
        """Возвращает имя модели эмбеддингов провайдера с учётом override."""
        if override:
            return override
        if provider == "openai":
            return self.cfg.openai_embedding_model_name
        if provider == "openrouter":
            return self.cfg.openrouter_emb_model
        if provider == "mistral":
            return self.cfg.mistral_emb_model
        raise ValueError(f"Неподдерживаемый провайдер: {provider}")

    def create_chat(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Создаёт чат-клиент LangChain для выбранного провайдера.

        Args:
            model: Имя модели (если None — берётся из настроек).
            api_key: Ключ API (если None — берётся из настроек).
            **kwargs: Доп. параметры (temperature, и т.д.).

        Returns:
            ChatOpenAI (openai/openrouter) или ChatMistralAI (mistral).
        """
        self.log.info("start:create_chat провайдер=%s", self.provider)
        key = self._resolve_api_key(api_key)
        p = self.provider
        m = self._chat_model_for_provider(p, model)

        if p in ("openai", "openrouter"):
            timeout = build_httpx_timeout(
                connect_s=self.cfg.connect_timeout_s,
                request_s=self.cfg.request_timeout_s,
            )
            common = dict(model=m, api_key=key, timeout=timeout, max_retries=0, **kwargs)

            if p == "openai":
                self.log.debug("create_chat: OpenAI, model=%s", m)
                if self.cfg.openai_api_base:
                    common["base_url"] = self.cfg.openai_api_base
                return ChatOpenAI(**common)

            base_url = getattr(self.cfg, "openrouter_base_url", "https://openrouter.ai/api/v1")
            headers = openrouter_headers(
                getattr(self.cfg, "openrouter_referer", None),
                getattr(self.cfg, "openrouter_title", None),
            )
            self.log.debug("create_chat: OpenRouter, model=%s, base=%s", m, base_url)
            return ChatOpenAI(**common, base_url=base_url, default_headers=headers)

        if p == "mistral":
            # Mistral ждёт timeout как int секунд
            tout = int(self.cfg.request_timeout_s)
            self.log.debug("create_chat: Mistral, model=%s, timeout=%ss", m, tout)
            return ChatMistralAI(model=m, api_key=key, timeout=tout, max_retries=0, **kwargs)

        raise ValueError(f"Неподдерживаемый провайдер: {p}")

    def create_embeddings(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Создаёт Embeddings-клиент для выбранного провайдера.

        Args:
            model: Имя модели эмбеддингов (если None — из настроек).
            api_key: Ключ API (если None — из настроек).
            **kwargs: Доп. параметры.

        Returns:
            OpenAIEmbeddings (openai/openrouter) или MistralAIEmbeddings (mistral).
        """
        self.log.info("start:create_embeddings провайдер=%s", self.provider)
        key = self._resolve_api_key(api_key)
        p = self.provider
        m = self._emb_model_for_provider(p, model)

        if p in ("openai", "openrouter"):
            params = dict(
                model=m,
                api_key=key,
                timeout=self.cfg.request_timeout_s,
                max_retries=0,
                **kwargs,
            )
            if p == "openrouter":
                base_url = getattr(self.cfg, "openrouter_base_url", "https://openrouter.ai/api/v1")
                params.update(
                    base_url=base_url,
                    default_headers=openrouter_headers(
                        getattr(self.cfg, "openrouter_referer", None),
                        getattr(self.cfg, "openrouter_title", None),
                    ),
                )
                self.log.debug("create_embeddings: OpenRouter, model=%s, base=%s", m, base_url)
            else:
                self.log.debug("create_embeddings: OpenAI, model=%s", m)
                if self.cfg.openai_api_base:
                    params["base_url"] = self.cfg.openai_api_base
            return OpenAIEmbeddings(**params)

        if p == "mistral":
            tout = int(self.cfg.request_timeout_s)
            self.log.debug("create_embeddings: Mistral, model=%s, timeout=%ss", m, tout)
            return MistralAIEmbeddings(model=m, api_key=key, timeout=tout, max_retries=0, **kwargs)

        raise ValueError(f"Неподдерживаемый провайдер: {p}")

    # ------------------------- публичные операции -------------------------

    def validate_api_key(self, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Делает минимальный вызов к чату и проверяет, что ключ «живой».

        Args:
            api_key: Явный ключ API (иначе из настроек).

        Returns:
            (ok, reason) — флаг успеха и краткая причина ошибки/успеха.
        """
        self.log.info("start:validate_api_key провайдер=%s", self.provider)
        key = self._resolve_api_key(api_key)
        if not key:
            return False, "missing"

        p = self.provider
        model = self._chat_model_for_provider(p, None)
        chat = self.create_chat(model=model, api_key=api_key, temperature=0.0)

        t0 = time.perf_counter()
        try:
            def _fn():
                return chat.invoke([HumanMessage(content="ping")])

            out = self._call_with_retry("validate_api_key", _fn)
            dt = (time.perf_counter() - t0) * 1000
            ok = bool(getattr(out, "content", None))
            self.log.info("validate_api_key: провайдер=%s, ок=%s, время=%.1f мс", p, ok, dt)
            return (True, "live_ok") if ok else (False, "live_failed_empty_response")
        except Exception as e:
            self.log.error("validate_api_key: ошибка %s", repr(e))
            return False, f"live_error:{type(e).__name__}:{e}"

    def generate(
        self,
        texts: Sequence[str],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Батч-генерация ответов: список промптов → список текстов.

        Args:
            texts: Список входных сообщений.
            model: Имя модели (если None — из настроек).
            api_key: Ключ API (если None — из настроек).
            **kwargs: Доп. параметры клиента (например, temperature).

        Returns:
            Список строк той же длины, что `texts`. При ошибках — пустые строки.
        """
        self.log.info("start:generate провайдер=%s, N=%d", self.provider, len(texts or []))
        if not texts:
            return []

        ok, reason = self.validate_api_key(api_key=api_key)
        if not ok:
            self.log.warning("generate: пропущено из-за ключа (%s)", reason)
            return ["" for _ in texts]

        chat = self.create_chat(model=model, api_key=api_key, **kwargs)
        results: List[str] = []

        for idx, t in enumerate(texts, 1):
            self.log.debug("generate: item %d/%d, prompt_len=%d", idx, len(texts), len(t or ""))

            def _fn():
                return chat.invoke([HumanMessage(content=t)])

            try:
                out = self._call_with_retry("generate", _fn)
                results.append(getattr(out, "content", "") or "")
            except Exception as e:
                self.log.error("generate: item %d ошибка %s", idx, repr(e))
                results.append("")

        self.log.info("generate: завершено провайдер=%s", self.provider)
        return results

    def embed(
        self,
        texts: Sequence[str],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Батч-эмбеддинг: список строк → матрица эмбеддингов.

        Args:
            texts: Список строк.
            model: Имя модели эмбеддингов (если None — из настроек).
            api_key: Ключ API (если None — из настроек).
            **kwargs: Доп. параметры клиента эмбеддингов.

        Returns:
            Список векторов; при ошибке в чанке — пустые векторы на его месте.
        """
        self.log.info("start:embed провайдер=%s, N=%d", self.provider, len(texts or []))
        if not texts:
            return []

        ok, reason = self.validate_api_key(api_key=api_key)
        if not ok:
            self.log.warning("embed: пропущено из-за ключа (%s)", reason)
            return [[] for _ in texts]

        emb = self.create_embeddings(model=model, api_key=api_key, **kwargs)
        batch = self.cfg.emb_batch_size
        total = len(texts)
        vectors: List[List[float]] = []

        for start in range(0, total, batch):
            end = min(start + batch, total)
            chunk = list(texts[start:end])
            self.log.debug("embed: chunk %d..%d", start, end)

            def _fn():
                return emb.embed_documents(chunk)

            try:
                part = self._call_with_retry("embed", _fn)
                vectors.extend(part)
            except Exception as e:
                self.log.error("embed: chunk %d..%d ошибка %s", start, end, repr(e))
                vectors.extend([[] for _ in chunk])

        self.log.info("embed: завершено провайдер=%s", self.provider)
        return vectors


if __name__ == "__main__":
    cfg = get_settings()
    client = LLMClient(provider=cfg.default_provider)
    print("validate_api_key:", client.validate_api_key())
    print("generate:", client.generate(["ping"], temperature=0.0))
    print("embed:", [len(v) for v in client.embed(["hello", "world"])])
