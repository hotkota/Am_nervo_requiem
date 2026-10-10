import redis
import discord
from discord.ext import commands


r = redis.Redis(
    db = 1,
    charset = "utf-8",
    decode_responses = True,
)


def tag_add(id: str, content: str) -> None:
    r.set(id, content)


def tag_get(id: str) -> str:
    return r.get(id)


class Tags(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        res = tag_get(message.content)
        if res is not None:
            await message.channel.send(str(res.strip()))

    @commands.command(name = "tag", aliases = ["тег"])
    async def tags(self, ctx):
        raw = ctx.message.content[4:].strip()

        if raw.startswith("создать"):
            if "|" not in raw:
                await ctx.message.reply(
                    "Формат: `.тег создать <имя>|<текст>`\n"
                    "Пример: `.тег создать привет|Привет, мир!`"
                )
                return

            content = raw.split("|", maxsplit = 1)
            name = content[0].strip()[len("создать"):].strip()
            text = content[1].strip()

            if not name:
                await ctx.message.reply("Укажи имя тега: `.тег создать <имя>|<текст>`")
                return

            tag_add(name, text)
            await ctx.message.add_reaction("✅")
            return

        name = raw.split("|", maxsplit = 1)[0].strip()
        res = tag_get(name)
        if res is not None:
            await ctx.message.channel.send(str(res.strip()))
        else:
            await ctx.message.reply(f"Тег `{name}` не найден")