import os
import re
import time
import logging
import asyncio
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
            f"max_new: {max_new_tokens}"
        )

        # --- Генерация ---
        start = time.perf_counter()
        result = self.llm.create_chat_completion(
            messages = messages,
            max_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            repeat_penalty = repetition_penalty,
            stop = ["<|im_end|>", "<|im_start|>"]
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

    async def generate_response_async(self, *args, **kwargs) -> str:
        """Асинхронная обёртка"""
        return await asyncio.to_thread(self.generate_response, *args, **kwargs)