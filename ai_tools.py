# ai_tools.py
# Инструменты (function calling) для Qwen-чата: схема OpenAI-совместимого
# формата + реализация поиска в интернете через DuckDuckGo (ddgs).

import json
import logging
from typing import Any, Callable

from ddgs import DDGS

logger = logging.getLogger(__name__)

# Максимум результатов выдачи на один запрос поиска
MAX_SEARCH_RESULTS = 5
# Сколько символов забираем из snippet'а результата, чтобы не раздувать контекст
SNIPPET_LIMIT = 300


def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """
    Ищет информацию в интернете через DuckDuckGo и возвращает список
    результатов в виде текста (заголовок / URL / фрагмент страницы).

    :param query: поисковый запрос
    :param max_results: количество результатов (1..10)
    :return: отформатированный текст результатов или сообщение об ошибке
    """
    max_results = max(1, min(int(max_results), 10))
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.exception("Ошибка при поиске в интернете")
        return f"Поиск не удался: {e}"

    if not results:
        return f"По запросу «{query}» ничего не найдено."

    lines = [f"Результаты поиска по запросу «{query}»:", ""]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "Без названия").strip()
        href = (r.get("href") or r.get("url") or "").strip()
        body = (r.get("body") or "").strip().replace("\n", " ")
        if len(body) > SNIPPET_LIMIT:
            body = body[:SNIPPET_LIMIT] + "…"
        lines.append(f"{i}. {title}\n   URL: {href}\n   Фрагмент: {body}")
    return "\n".join(lines)


# Реестр: имя инструмента -> python-функция, его реализующая
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "web_search": web_search,
}

# Описание инструментов в формате tools (OpenAI-совместимом), который
# понимает chat template Qwen (tool_call / tool_response).
TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Поиск в интернете через DuckDuckGo. Используй этот инструмент, "
                "когда у тебя нет актуальной информации, тебя спрашивают о свежих "
                "новостях, ценах, погоде, событиях или когда ты не уверен в фактах. "
                "Твои знания ограничены обучающими данными — для всего "
                "остального ищи информацию в сети."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (короткий и конкретный).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": f"Число результатов (1..10), по умолчанию {MAX_SEARCH_RESULTS}.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """
    Пытается разобрать ответ модели как список вызовов инструментов.

    Qwen2.5 в чате выдаёт их так:
        {"name": "web_search", "arguments": {"query": "...", "max_results": 5}}

    Также поддерживается вариант с ```json блоком и одиночный вызов без обёртки.

    :return: список словарей [{"name": ..., "arguments": {...}}, ...]
             или None, если текст не похож на вызов инструмента.
    """
    s = text.strip()
    if not s:
        return None

    # вытаскиваем JSON из markdown-блока, если он есть
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()

    # одиночный объект {"name": ..., "arguments": ...}
    def _as_call(obj: Any) -> dict[str, Any] | None:
        if isinstance(obj, dict) and obj.get("name") in TOOL_FUNCTIONS:
            args = obj.get("arguments", {})
            if isinstance(args, str):  # иногда arguments приходит строкой
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": obj["name"], "arguments": args}
        return None

    candidate = _as_call(_try_json(s))
    if candidate:
        return [candidate]

    # список вызовов [{...}, {...}]
    parsed = _try_json(s)
    if isinstance(parsed, list):
        calls = [c for c in (_as_call(item) for item in parsed) if c]
        if calls:
            return calls

    return None


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def execute_tool_call(call: dict[str, Any]) -> str:
    """
    Выполняет разобранный вызов инструмента и возвращает его результат
    в виде строки (её отправляют модели как tool_response).
    """
    name = call.get("name")
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Неизвестный инструмент: {name}"
    try:
        return fn(**call.get("arguments", {}))
    except TypeError as e:
        return f"Некорректные аргументы инструмента {name}: {e}"
    except Exception as e:
        logger.exception("Ошибка выполнения инструмента %s", name)
        return f"Инструмент {name} завершился с ошибкой: {e}"
