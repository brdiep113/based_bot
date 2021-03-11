import torch
import torch.nn as nn
from pytorch_model.model import BasedBot
from process_tweets import csv_to_df, process_text, pytorch_preprocess
from pytorch_model.preprocess_data import setup_dataloaders
from pytorch_model.train import train
import argparse

if __name__ == '__main__':
    # Load data
    df_tweets = csv_to_df('tweet_data/tweets_users_with_party.csv')

    # Set up data loaders
    train_loader, valid_loader, test_loader, vocabulary = setup_dataloaders(df_tweets, batch_size=50)

    # Set up BasedBot

    BasedBot = BasedBot(vocabulary, train_loader, valid_loader, test_loader)

    # Train on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BasedBot.model.to(device)

    val_losses = train(BasedBot.model, train_loader=BasedBot.train_loader, val_loader=BasedBot.valid_loader, epochs=5)

