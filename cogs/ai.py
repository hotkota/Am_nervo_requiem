import os
import asyncio
import discord
from discord.ext import commands

from qwen_model import QwenChat


SYSTEM_PROMPT = (
    "Всегда отвечай кратко, по делу, на русском языке. "
    "Учитывай предыдущие сообщения в этом диалоге. "
    "Если пользователь ранее называл своё имя или другую информацию о себе — используй её в ответах. "
    "Тебя зовут Мруа бот. Используй это имя. "
)


class AI(commands.Cog):
    def __init__(self, bot, chat: QwenChat):
        self.bot = bot
        self.chat = chat
        self.lock = asyncio.Lock()

        # История: channel_id -> список сообщений
        self.histories: dict[int, list] = {}

        # Сколько последних пар «user-assistant» хранить
        self.max_history = int(os.getenv("MAX_HISTORY", 5))

    async def _send_long(self, ctx, text: str):
        if not text.strip():
            text = "(пустой ответ)"
        for i in range(0, len(text), 1900):
            await ctx.send(text[i:i + 1900])

    def _get_history(self, channel_id: int) -> list:
        """Возвращает историю канала или создаёт новую с system-промптом."""
        if channel_id not in self.histories:
            self.histories[channel_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.histories[channel_id]

    def _trim_history(self, history: list):
        """Оставляет system + последние N пар user/assistant."""
        # system всегда на позиции 0
        # после него идут пары user/assistant — по 2 сообщения на пару
        max_len = 1 + self.max_history * 2
        if len(history) > max_len:
            # Сохраняем system, обрезаем всё лишнее с начала диалога
            del history[1:len(history) - self.max_history * 2]

    @commands.command(name = "mrua", aliases = ["мруа"])
    async def ask(self, ctx, *, prompt: str = ""):
        # Команда очистки истории
        if prompt.strip().lower() in ("сброс", "reset", "clear"):
            self.histories.pop(ctx.channel.id, None)
            await ctx.message.reply("История диалога очищена.")
            return

        if not prompt.strip():
            await ctx.message.reply("Не вижу вопроса")
            return

        # Берём историю канала
        history = self._get_history(ctx.channel.id)

        # Добавляем текущий вопрос
        history.append({"role": "user", "content": prompt})

        log_context = f"user={ctx.author.display_name}({ctx.author.id})"

        async with ctx.message.channel.typing():
            async with self.lock:
                reply = await self.chat.generate_response_async(
                    history,
                    max_new_tokens = 100,
                    log_context = log_context,
                )

        # Добавляем ответ ассистента в историю
        history.append({"role": "assistant", "content": reply})

        # Обрезаем до последних N пар
        self._trim_history(history)

        await self._send_long(ctx, reply)