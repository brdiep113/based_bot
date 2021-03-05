import json
import pandas as pd
import test.test__osx_support
from bs4 import BeautifulSoup
import os
import requests

def process_tweets(fpath):
    """ Processes tweets into a Pandas DataFrame

    Args:
        fpath (string): Relative filepath to tweet JSON
        
    Returns:
        df_tweet (DataFrame) DataFrame with tweet text and user_id information
    """
    
    path = os.path.relpath(fpath)
    
    tweets_file = open(path)
    
    line = tweets_file.readline()
    
    tweet_dict_list = []
    
    while line:
        tweet_dict = json.loads(line)
        clean_tweets_dict = {
            'text' : tweet_dict['text'],
            'user_id' : tweet_dict['user_id']
        }
        
        tweet_dict_list.append(clean_tweets_dict)
        
        line = tweets_file.readline()
    
    return pd.DataFrame(tweet_dict_list)

def process_users(fpath):
    """ Processes twitter users into a DataFrame

    Args:
        fpath (string): Relative filepath to users JSON
        
    Returns:
        df_user (DataFrame): DataFrame with user id and name information
    """
    
    path = os.path.relpath(fpath)
    
    users_file = open(path)
    
    line = users_file.readline()
    
    user_dict_list = []
    
    while line:
        user_dict = json.loads(line)
        clean_user_dict = {
            'user_id' : user_dict['id'],
            'name' : user_dict['name']
        }
        
        user_dict_list.append(clean_user_dict)
        
        line = users_file.readline()
    
    return pd.DataFrame(user_dict_list)
    
def join_tweets_users(df_tweet, df_user):
    """ Joins a dataframe of tweets with a dataframe of users by matching user ids to tweets

    Args:
        df_tweet (DataFrame): A dataframe containing tweet text with user ids
        df_user (DataFrame): A dataframe containing user name and id
        
    Returns:
        df_tweet_user (DataFrame): A dataframe containing tweet text, user id, and user name
    """
    
    df_tweet_user = df_tweet.merge(df_user, on='user_id', how='left')
    
    return df_tweet_user

def json_to_df(fpath):
    """ Parses a JSON to a dataframe

    Args:
        fpath (string): Relative filepath to a JSON representation of a dataframe
        
    Returns:
        df (DataFrame): DataFrame representation of JSON data found at filepath
    """
    
    path = os.path.relpath(fpath)
    
    return pd.read_json(fpath)

def csv_to_df(fpath):
    """ Parses a JSON to a dataframe

    Args:
        fpath (string): Relative filepath to a CSV representation of a dataframe
        
    Returns:
        df (DataFrame): DataFrame representation of CSV data found at filepath
    """
    
    path = os.path.relpath(fpath)
    
    return pd.read_csv(fpath)

def get_political_party(name):
    """ Scrapes Wikipedia for the political party of the given person, defaults Independent politicians to their last mainstream political party

    Args:
        name (string): The name of a politician
        
    Returns:
        party (string): The name of the politician's political party
    """
        
    url = f'https://en.wikipedia.org/wiki.php?search={name}'
    
    html = requests.get(url).text
    
    soup = BeautifulSoup(html, features='lxml')
    
    politican_info = soup.findAll('table', attrs={'class' : 'infobox vcard'})
    
    if not politican_info:
        return 'Unknown'
    
    dem_party = politican_info[0]('a', attrs={'title' : 'Democratic Party (United States)'})
    
    rep_party = politican_info[0]('a', attrs={'title' : 'Republican Party (United States)'})
    
    if dem_party:
        return 'Democrat'
    
    elif rep_party:
        return 'Republican'
    
    else:
        return 'Independent'

def add_political_party(df_politician):
    """ Adds the political party of politicians to a dataframe of politician data

    Args:
        df_politician (DataFrame): DataFrame with politician information
        
    Returns:
        df_politician_party (DataFrame) DataFrame with politician information, including political party
    """
    
    parties_list = []
    
    for i in range(len(df_politician)):
        politician = df_politician.loc[i, 'name']
        parties_list.append(get_political_party(politician))
        
    df_politician['party'] = parties_list
    
    return df_politician

if __name__ == '__main__':
    df_tweet = json_to_df('tweet_data/processed/clean_tweets.json')
    df_user = csv_to_df('tweet_data/processed/users_with_party.csv')
    df_tweet_user = join_tweets_users(df_tweet, df_user)
    df_tweet_user.to_csv('tweet_data/processed/tweets_users_with_party.csv')