import json
import numpy as np
import pandas as pd
from string import punctuation
from collections import Counter
from bs4 import BeautifulSoup
import os
import requests
import re


def process_politician_db():
    member_info = pd.read_csv('tweet_data/parliamentarians/full_member_info.csv', encoding='utf-16',
                              engine='python')
    tweets_data = pd.DataFrame()

    with open('tweet_data/parliamentarians/all_tweet_ids.jsonl', encoding='utf-8') as f:
        i = 0
        for line in f:
            data = json.loads(line)
            if data['lang'] == 'en':
                print(i)
                reduced = {'full_text':data['full_text'], 'id':data['id']}
                tweets_data = tweets_data.append(reduced, ignore_index=True)
                i += 1

    tweets_data.rename(columns={'id':'uid'}, inplace=True)
    member_info_filtered = member_info[['country', 'name', 'party', 'left_right', 'uid']].copy()

    df = pd.merge(member_info_filtered, tweets_data, how='left', on='uid')

    return df


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
    
    return pd.read_json(path)

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

def process_text(df):
    """

    :rtype: object
    """
    features = df['text']
    labels = df['party']
    processed_features = []
    for sentence in range(0, len(features)):
        # Remove all the special characters
        processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

        # remove all single characters
        processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

        # Remove single characters from the start
        processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

        # Substituting multiple spaces with single space
        processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
        # Converting to Lowercase
        processed_feature = processed_feature.lower()
        processed_features.append(processed_feature)
        print(processed_feature)
    return processed_features, labels


def pytorch_preprocess(df):
    tweets = df['text']
    labels = df['party']

    processed_tweets = []
    for tweet in range(len(tweets)):
        processed_tweet = tweet.lower()
        processed_tweet = processed_tweet.translate(str.maketrans('', '', processed_tweet.punctuation))
        processed_tweets.append(processed_tweet)

    all_text2 = ' '.join(processed_tweets)

    # create a list of words
    words = all_text2.split()  # Count all the words using Counter Method
    count_words = Counter(words)

    total_words = len(words)
    sorted_words = count_words.most_common(total_words)
    vocab_to_int = {w:i+1 for i, (w, c) in enumerate(sorted_words)}

    tweets_int = []
    for tweet in processed_tweets:
        t = [vocab_to_int[w] for w in tweet.split()]
        tweets_int.append(t)

    encoded_labels = [1 if label == 'Democrat' else 0 for label in labels]
    encoded_labels = np.array(encoded_labels)

    return pad_features(tweets_int), encoded_labels, total_words


def pad_features(tweets_int, seq_len=250):

    features = np.zeros((len(tweets_int), seq_len), dtype=int)

    for i, tweet in enumerate(tweets_int):
        tweet_len = len(tweet)
        if tweet_len <= seq_len:
            zeroes = list(np.zeros(seq_len - tweet_len))
            new = zeroes + tweet
        elif tweet_len > seq_len:
            new = tweet[:seq_len]

        features[i,:] = np.array(new)

    return features


def train_val_test_split(features, labels, split_frac=0.8):

    length = len(features)
    X_train = features[:int(split_frac * length)]
    y_train = labels[:int(split_frac * length)]
    
    remaining_x = features[int(split_frac * length):]
    remaining_y = labels[int(split_frac * length):]
    
    X_val = remaining_x[:int(len(remaining_x) * 0.5)]
    y_val = remaining_y[:int(len(remaining_y) * 0.5)]

    X_test = remaining_x[int(len(remaining_x) * 0.5):]
    y_test = remaining_y[int(len(remaining_y) * 0.5):]
    
    return X_train, y_train, X_val, y_val, X_test, y_test



if __name__ == '__main__':
    #df = process_politician_db('a')
    pass