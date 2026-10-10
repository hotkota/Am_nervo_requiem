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

bot = commands.Bot(command_prefix = ".", intents=intents)


@bot.event
async def on_ready():
    stream_url = os.getenv("STREAM_URL", "https://www.twitch.tv/hotkota")
    await bot.change_presence(
        activity = discord.Streaming(
            name = "В мруа потоке",
            url = stream_url,
        ),
        status = discord.Status.online,
    )
    await bot.add_cog(Tags(bot))
    await bot.add_cog(AI(bot, chat))
    print(f'We have logged in as {bot.user}')


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
    bot.run(token)