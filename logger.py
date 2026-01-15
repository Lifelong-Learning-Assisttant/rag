"""
Логгер
"""

import logging
from settings import get_settings

_LEVELS = {
    "DEBUG":    logging.DEBUG,
    "INFO":     logging.INFO,
    "WARNING":  logging.WARNING,
    "ERROR":    logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

def _parse_level(name: str) -> int:
    """Строковый уровень -> logging level, по умолчанию INFO."""
    return _LEVELS.get((name or "INFO").strip().upper(), logging.INFO)

def get_logger(name: str = "app") -> logging.Logger:
    """
    Простой логгер:
    - Уровень берётся из settings.log_level (LLM_LOG_LEVEL).
    - Один StreamHandler, без дублей.
    - Формат: time | LEVEL | name | message.
    """
    cfg = get_settings()
    level = _parse_level(cfg.log_level)

    log = logging.getLogger(name)
    log.setLevel(level)          # обновляем уровень при каждом вызове
    log.propagate = False

    if not log.handlers:
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h = logging.StreamHandler()   # stderr по умолчанию
        h.setLevel(level)
        h.setFormatter(fmt)
        log.addHandler(h)
    else:
        # синхронизируем уровень уже существующих хендлеров с settings
        for h in log.handlers:
            h.setLevel(level)

    return log


if __name__ == "__main__":
    # Пример запуска: LLM_LOG_LEVEL=DEBUG uv run logger.py
    log = get_logger("demo")
    log.debug("debug msg (видно при LLM_LOG_LEVEL=DEBUG)")
    log.info("info msg")
    log.warning("warn msg")
