import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional, Tuple

import httpx
from httpx import HTTPStatusError
from tenacity import RetryError


def truncate(text: str, limit: int = 300) -> str:
    """
    Безопасно обрезает строку для логов.

    Args:
        text: Исходная строка.
        limit: Максимальная длина.

    Returns:
        str: Обрезанный текст с многоточием при необходимости.
    """
    return text if len(text) <= limit else (text[:limit] + "…")


def parse_retry_after(value: Optional[str]) -> Optional[int]:
    """
    Парсит заголовок Retry-After: секунды или HTTP-date → секунды.

    Args:
        value: Значение заголовка.

    Returns:
        Optional[int]: Секунды ожидания или None.
    """
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        pass
    try:
        dt = parsedate_to_datetime(value)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(max(0, (dt - datetime.now(timezone.utc)).total_seconds()))
    except Exception:
        return None


def extract_request_id_from_exc(exc: Exception) -> Optional[str]:
    """
    Возвращает X-Request-ID/Request-ID из HTTP-ответа, если есть.

    Args:
        exc: Исключение.

    Returns:
        Optional[str]: Идентификатор запроса.
    """
    if isinstance(exc, HTTPStatusError) and exc.response is not None:
        for k in ("x-request-id", "x-requestid", "request-id"):
            if k in exc.response.headers:
                return exc.response.headers.get(k)
    return None


def unwrap_http_exc(
    exc: Exception,
) -> Tuple[Exception, Optional[int], Optional[int], Optional[str], str]:
    """
    Разворачивает исключения до деталей HTTP: статус, Retry-After, request-id, тело.

    Args:
        exc: Исключение верхнего уровня (в т.ч. RetryError).

    Returns:
        (исх_исключение, status, retry_after_s, request_id, body_snippet)
    """
    # Если это RetryError — вытащим исходную ошибку
    if isinstance(exc, RetryError) and exc.last_attempt:
        try:
            inner = exc.last_attempt.exception()
            return unwrap_http_exc(inner)
        except Exception:
            pass

    status = None
    retry_after = None
    request_id = None
    body_snippet = ""

    if isinstance(exc, HTTPStatusError) and exc.response is not None:
        status = exc.response.status_code
        request_id = extract_request_id_from_exc(exc)
        retry_after = parse_retry_after(exc.response.headers.get("retry-after"))
        try:
            body_json = exc.response.json()
            body_snippet = truncate(json.dumps(body_json, ensure_ascii=False))
        except Exception:
            try:
                body_snippet = truncate(exc.response.text or "")
            except Exception:
                body_snippet = ""

    return exc, status, retry_after, request_id, body_snippet


def build_httpx_timeout(connect_s: float, request_s: float) -> httpx.Timeout:
    """
    Создаёт httpx.Timeout с равными read/write/pool таймаутами.

    Args:
        connect_s: Таймаут установления соединения.
        request_s: Таймаут чтения/записи/пула.

    Returns:
        httpx.Timeout: Объект таймаутов.
    """
    return httpx.Timeout(connect=connect_s, read=request_s, write=request_s, pool=request_s)


def openrouter_headers(referer: Optional[str], title: Optional[str]) -> dict:
    """
    Атрибуционные заголовки для OpenRouter.

    Args:
        referer: HTTP-Referer (URL/имя приложения).
        title: X-Title (человекочитаемое имя).

    Returns:
        dict: Заголовки (может быть пустым).
    """
    headers = {}
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers
