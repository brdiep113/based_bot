import discord
import json
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from discord.ext import commands
from utils import clean_text
import os

with open(os.path.relpath('config')) as config:
    config_dict = json.load(config)

intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

pickle_model = open("political_sentiment_analyzer.joblib", "rb")
classifier_model = joblib.load("political_sentiment_analyzer.joblib")
vectorizer = joblib.load("political_sentiment_vectorizer.joblib")

@bot.event
async def on_ready():
    """
    Notification that the bot has logged into the server
    """

    print('Hello world!')

@bot.command()
async def hottake(ctx, *args):
    """
    :param ctx: Discord bot Context
    """

    # server = bot.get_guild(config_dict['server_id'])
    channel = bot.get_channel(ctx.channel.id)
    text = " ".join(args)

    # TO DO: clean text before fitting to classifier

    # Vectorize text
    text = vectorizer.transform([text])

    prediction = classifier_model.predict(text)

    if prediction == "Democrat":
        message = "Based"
    else:
        message = "Cringe"

    await channel.send(message)

if __name__ == "__main__":
    bot.run(config_dict['api_key'])
