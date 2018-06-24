from src.features.process_tweet.clean_tweet import clean_tweet
from tqdm import tqdm

def main():
   
  
    # Clean tweets.
    for review in tqdm(review_test_sent_list): clean_tweet(review)



if __name__ == '__main__':
    main()