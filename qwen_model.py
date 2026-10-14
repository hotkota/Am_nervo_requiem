import os
import re
import time
import json
import logging
import asyncio
from typing import Any, Callable
from dotenv import load_dotenv
from llama_cpp import Llama

load_dotenv()

# --- Логирование ---
logger = logging.getLogger("qwen")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class QwenChat:
    """Обёртка над ИИ"""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int | None = None):
        self.model_path = model_path
        self.n_threads = n_threads or (os.cpu_count() or 4)

        logger.info(f"[QwenChat] Загрузка GGUF: {model_path}")
        logger.info(f"[QwenChat] Потоков: {self.n_threads}, контекст: {n_ctx}")

        self.llm = Llama(
            model_path = model_path,
            n_ctx = n_ctx,
            n_threads = self.n_threads,
            n_batch = 128,
            chat_format = "chatml",
            verbose = False
        )

        logger.info(f"[QwenChat] Модель загружена.")

    def warmup(self):
        """Холостой прогон"""
        self.llm.create_chat_completion(
            messages = [{"role": "user", "content": "warmup"}],
            max_tokens = 5,
        )
        logger.info("[QwenChat] Прогрев завершён.")

    def generate_response(
        self,
        messages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        log_context: str = "",
    ) -> str:
        """Синхронная генерация через llama.cpp."""

        # Значения из .env или дефолты
        max_new_tokens = max_new_tokens or int(os.getenv("MAX_NEW_TOKENS", 100))
        temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", 0.6))
        top_p = top_p if top_p is not None else float(os.getenv("TOP_P", 0.8))
        top_k = top_k if top_k is not None else int(os.getenv("TOP_K", 20))
        repetition_penalty = repetition_penalty if repetition_penalty is not None else float(os.getenv("REPETITION_PENALTY", 1.1))

        # Qwen3 переключает режим через /think и /no_think в промпте.
        # Копируем, чтобы не мутировать исходные messages.
        messages = [dict(m) for m in messages]
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = f"{msg['content']} /no_think"
                break

        sampling_kwargs = dict(
            max_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            repeat_penalty = repetition_penalty,
        )

        return self._generate_once(messages, sampling_kwargs, log_context)

    def _generate_once(self, messages: list[dict], sampling_kwargs: dict, log_context: str) -> str:
        """Один проход генерации через llama.cpp + метрики. Возвращает текст ответа."""

        # --- Метрики входа ---
        prompt_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break
        prompt_chars = len(prompt_text)

        logger.info(
            f"[START] {log_context} | "
            f"запрос: {prompt_chars} симв | "
            f"сообщений: {len(messages)} | "
            f"max_new: {sampling_kwargs.get('max_tokens')}"
        )

        # --- Генерация ---
        start = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages = messages,
            stop = ["<|im_end|>", "<|im_start|>"],
            **sampling_kwargs
        )
        elapsed = time.perf_counter() - start

        # --- Метрики выхода ---
        reply = result["choices"][0]["message"]["content"].strip()

        # Убираем блоки <think>...</think> и одиночные теги
        reply = re.sub(r"<think>.*?</think>", "", reply, flags = re.DOTALL).strip()
        reply = reply.replace("<think>", "").replace("</think>", "").strip()

        reply_chars = len(reply)

        usage = result.get("usage", {})
        new_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get("prompt_tokens", 0)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"[DONE]  {log_context} | "
            f"ответ: {reply_chars} симв | "
            f"вход: {input_tokens} ток | "
            f"токенов: {new_tokens} | "
            f"время: {elapsed:.2f} с | "
            f"скорость: {tokens_per_sec:.2f} ток/с"
        )

        return reply

    def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_executor,
        tool_parser=None,
        max_tool_rounds: int = 3,
        max_new_tokens: int | None = None,
        log_context: str = "",
    ) -> tuple[str, list[dict]]:
        """
        Генерация с инструментами (function calling) в стиле ReAct:

        1. Модель получает сообщения + описание инструментов в system-промпте.
        2. Если её ответ разбирается как вызов инструмента — мы исполняем
           его (tool_executor), добавляем в диалог role=tool сообщение с
           результатом и просим модель ответить заново, уже с данными.
        3. Цикл повторяется до max_tool_rounds раз, пока модель не даст
           обычный текстовый ответ.

        :return: (итоговый ответ, новые сообщения для сохранения в историю)
        """
        if tool_parser is None:
            from ai_tools import parse_tool_calls
            tool_parser = parse_tool_calls

        max_new_tokens = max_new_tokens or int(os.getenv("MAX_NEW_TOKENS", 100))

        # Копия, чтобы не мутировать историю вызывающей стороны
        convo = [dict(m) for m in messages]

        # Вставляем инструкцию по использованию инструментов в system-промпт.
        tools_note = (
            "\n\nДОСТУПНЫЕ ИНСТРУМЕНТЫ (tools):\n"
            + json.dumps(tools, ensure_ascii=False)
            + "\n\nЕсли тебе нужна актуальная информация из интернета — "
            "не отвечай сразу, а сначала верни СТРОГО JSON-вызов вида "
            '{"name": "web_search", "arguments": {"query": "..."}}. '
            "После получения результата поиска (роль tool) дай финальный "
            "ответ пользователю обычным текстом. Если инструмент не нужен — "
            "просто ответь текстом."
        )
        if convo and convo[0].get("role") == "system":
            convo[0] = dict(convo[0])
            convo[0]["content"] = convo[0]["content"] + tools_note
        else:
            convo.insert(0, {"role": "system", "content": "Ты ассистент с доступом к инструментам." + tools_note})

        # /no_think — в последнее пользовательское сообщение текущего запроса
        for msg in reversed(convo):
            if msg.get("role") == "user":
                if "/no_think" not in msg["content"]:
                    msg["content"] = f"{msg['content']} /no_think"
                break

        n_original = len(messages)  # исходные сообщения уже есть в истории
        rounds_used = 0
        reply = ""

        while True:
            reply = self._generate_once(
                convo,
                dict(max_tokens = max_new_tokens),
                f"{log_context} round={rounds_used}".strip(),
            )

            calls = tool_parser(reply) if reply else None

            if not calls or rounds_used >= max_tool_rounds:
                if calls:
                    logger.warning(f"[TOOLS] {log_context}: лимит раундов исчерпан, отбрасываем вызовы")
                convo.append({"role": "assistant", "content": reply})
                break

            rounds_used += 1
            convo.append({"role": "assistant", "content": reply})
            for call in calls:
                logger.info(f"[TOOLS] {log_context}: вызов {call['name']} args={call['arguments']}")
                observation = tool_executor(call)
                convo.append({
                    "role": "tool",
                    "name": call["name"],
                    "content": observation,
                })

        # Всё, что появилось сверх исходной истории (кроме system) — новые сообщения
        new_messages = [m for m in convo[n_original:] if m.get("role") != "system"]
        return reply, new_messages

    async def generate_response_async(self, *args, **kwargs) -> str:
        """Асинхронная обёртка"""
        return await asyncio.to_thread(self.generate_response, *args, **kwargs)

    async def generate_with_tools_async(self, *args, **kwargs) -> tuple[str, list[dict]]:
        """Асинхронная обёртка для генерации с инструментами."""
        return await asyncio.to_thread(self.generate_with_tools, *args, **kwargs)
