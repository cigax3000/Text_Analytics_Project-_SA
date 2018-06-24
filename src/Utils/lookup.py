def get_review_by_review_id(review_set, review_id):
    """
    Retrieves tweet, based on unique tweet id.
    Args:
        tweet_set: A set of tweets.
        tweet_id: A unique tweet id.

    Returns:
        A tweet with the defined tweet id.
    """
    for review in reviews_set:
        if review.reviewer_id == reviewer_id:
            return review
