import os
import discord
from discord.ext import commands
from dotenv import load_dotenv

from qwen_model import QwenChat
from cogs.tags import Tags
from cogs.ai import AI

load_dotenv()


intents = discord.Intents.default()
intents.message_content = True

class MruaBot(commands.Bot):
    def __init__(self, chat: QwenChat):
        super().__init__(command_prefix=".", intents=intents)
        self.chat = chat

    async def setup_hook(self):
        # вызывается фреймворком один раз перед подключением к шлюзу —
        # рекомендуемый способ регистрации когов вместо устаревшего add_cog в on_ready
        await self.add_cog(Tags(self))
        await self.add_cog(AI(self, self.chat))

    async def on_ready(self):
        stream_url = os.getenv("STREAM_URL", "https://www.twitch.tv/hotkota")
        await self.change_presence(
            activity=discord.Streaming(
                name="В мруа потоке",
                url=stream_url,
            ),
            status=discord.Status.online,
        )
        print(f'We have logged in as {self.user}')



if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise SystemExit("DISCORD_TOKEN не задан в .env")

    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise SystemExit("MODEL_PATH не задан в .env")

    n_threads = int(os.getenv("NUM_THREADS", os.cpu_count() or 4))
    n_ctx = int(os.getenv("N_CTX", 2048))

    print(f"Загружаю {model_path}...")
    chat = QwenChat(model_path = model_path, n_ctx = n_ctx, n_threads = n_threads)
    chat.warmup()
    print("Модель готова. Запускаю бота...")
    bot = MruaBot(chat)
    bot.run(token)