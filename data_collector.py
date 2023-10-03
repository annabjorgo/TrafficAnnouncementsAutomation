from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import requests
import tweepy
from tweepy import Tweet
import csv

csv_name = f"output-{date.today().day}-{date.today().month}-{datetime.today().hour}.csv"
final_date = datetime(2023, 9, 27)
nrk_user_name = "NrkTrafikk"
nrk_user_id = "20629858"
fields = ['author_id', "tweet_id", "text", "referenced_tweets", "referenced_type", "referenced_user", "created_at"]

end_point = f"https://api.twitter.com/2/users/{nrk_user_id}/tweets"

client = tweepy.Client(consumer_key="",
                       consumer_secret="",
                       bearer_token="",
                       access_token="",
                       access_token_secret="")


def get_user_id() -> str:
    nrk_user_id = client.get_user(username=nrk_user_name).data.id
    print(nrk_user_id)
    return str(nrk_user_id)


def collect_timeline_tweets(iterations: int, number_of_tweets: int, start_time=None):
    data_list = []
    pageination_token = None
    for _ in range(0, iterations):
        try:
            res = client.get_users_tweets(nrk_user_id,
                                      tweet_fields=tweepy.PUBLIC_TWEET_FIELDS,
                                      max_results=number_of_tweets,
                                      pagination_token=pageination_token,
                                      start_time=start_time,
                                      )
            start_time=None
            extract_data(data_list, res)
        except Exception as e:
            print(e)
            save_csv(data_list)
        pageination_token = res.meta['next_token']
    return data_list


def save_csv(list_dict, csv_name=csv_name):
    with open(csv_name, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(list_dict)


def extract_data(data_list, res):
    for tweet in res.data:
        data_list.append(
            {fields[0]: tweet.author_id,
             fields[1]: tweet.id,
             fields[2]: tweet.text,
             fields[3]: "" if tweet.referenced_tweets is None else tweet.referenced_tweets[0].id,
             fields[4]: "" if tweet.referenced_tweets is None else tweet.referenced_tweets[0].type,
             fields[5]: "" if tweet.in_reply_to_user_id is None else tweet.in_reply_to_user_id,
             fields[6]: tweet.created_at
             })


data = collect_timeline_tweets(iterations=30, number_of_tweets=100, start_time=final_date)
save_csv(data)

