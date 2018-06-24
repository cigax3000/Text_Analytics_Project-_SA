from tabulate import tabulate


def format_review_table(tweets):
    """
    Formats the review in a tabular style.
    Args:
        review: A list or set of review objects
    Returns:
        A string formatted table of tweets
    """
    print_list = [_convert_tweet_to_list(review) for review in review]
    return tabulate(print_list, headers=["Date", "Retweet Count", "Favorite Count", "Text"])


def format_mentions_table(mentions):
    """
    Formats the mentions in a tabular style.
    Args:
        mentions: A dictionary of mentions

    Returns:
        A string formatted table of mentions
    """
    return tabulate(mentions, headers=["Tag", "Count"])


def _convert_tweet_to_list(tweet):
    """
    Converts a week_3.tweet to a list object
    Args:
        tweet: A week_3.tweet object

    Returns:
        A list with week_3.tweet information
    """

    return [tweet.created_at, tweet.retweet_count, tweet.favorite_count, tweet.text]