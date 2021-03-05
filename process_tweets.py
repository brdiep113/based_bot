import json
import pandas as pd
import test.test__osx_support
import os

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

if __name__ == '__main__':
    df_tweet = json_to_df('tweet_data/processed/clean_tweets.json')
    df_user = json_to_df('tweet_data/processed/clean_users.json')
    df_tweet_user = join_tweets_users(df_tweet, df_user)
    df_tweet_user.to_csv('tweet_data/processed/clean_tweet_users.csv')