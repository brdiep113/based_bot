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
            'id' : user_dict['id'],
            'name' : user_dict['name']
        }
        
        user_dict_list.append(clean_user_dict)
        
        line = users_file.readline()
    
    return pd.DataFrame(user_dict_list)

def save_df(df, fname):
    """ Saves the given dataframe at a given relative file path as a JSON

    Args:
        df (DataFrame): A Pandas DataFrame
    
        fname (string): A string relative file path
    """

    df.to_json(fname)

if __name__ == '__main__':
    df_tweet = process_users('tweet_data/users.json')
    save_df(df_tweet, 'tweet_data/processed/clean_users.json')