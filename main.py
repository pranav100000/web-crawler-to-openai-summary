import crawler.web_crawler as crawler
import llm_api_client.open_ai_client as open_ai_clt
from dotenv import load_dotenv
from collections import namedtuple
import datetime
import matplotlib.pyplot as plt


class WebCrawlerToOpenAIClient:
    def __init__(self):
        self.crawler = crawler.WebCrawler()
        self.open_ai_client = open_ai_clt.OpenAIClient()
        
    def scrape_and_rate_query(self, query, num_pages, start_n_days_ago, end_n_days_ago):
        summaries_and_urls = self.crawler.crawl_query(query, num_pages, start_n_days_ago, end_n_days_ago)
        summaries = [x[1] for x in summaries_and_urls]
        formatted_summaries = "\n\n".join(summaries)
        sentiment_score, explanation, summary = self.open_ai_client.get_sentiment_and_summary(query, formatted_summaries)
        print("\n\n________")
        print("Sentiment Score: " + str(sentiment_score) + "/100")
        print("\n")
        print("Explanation: " + explanation)
        print("\n")
        print("Summary: " + summary)
        print("________\n\n")
        return sentiment_score, explanation, summary, summaries_and_urls
    

load_dotenv()

query = input("Enter a query: ")
start_n_days_ago = int(input("Enter the number of days ago to start scraping from: "))
end_n_days_ago = int(input("Enter the number of days ago to end scraping from: "))
num_intervals = int(input("Enter the number of data points: "))
num_pages_to_scrape = int(input("Enter the number of pages to scrape per data point: "))


web_crawler_to_open_ai_client = WebCrawlerToOpenAIClient()
DateInterval = namedtuple("DateInterval", ["start", "end"])
date_intervals, sentiment_scores, explanations, summaries, all_summaries_and_urls = [], [], [], [], []
interval_size = (start_n_days_ago - end_n_days_ago) // num_intervals

for start_interval in range(start_n_days_ago, end_n_days_ago, -interval_size):
    if start_interval - interval_size < 0:
        break
    interval = DateInterval(start_interval, start_interval - interval_size)
    sentiment_score, explanation, summary, summaries_and_urls = web_crawler_to_open_ai_client.scrape_and_rate_query(query, num_pages_to_scrape, interval.start, interval.end)
    date_intervals.append(interval)
    sentiment_scores.append(sentiment_score)
    explanations.append(explanation)
    summaries.append(summary)
    all_summaries_and_urls.append(summaries_and_urls)

string_date_intervals = []
for i, interval in enumerate(date_intervals):
    curr_time = datetime.datetime.now()
    start_date, end_date = curr_time - datetime.timedelta(days=interval.start), curr_time - datetime.timedelta(days=interval.end)
    start_date, end_date = start_date.strftime("%m-%d-%Y"), end_date.strftime("%m-%d-%Y")
    string_date_intervals.append((start_date + " to " + end_date))
    print("________")
    print("Interval " + str(i+1) + ":")
    print("Start: " + start_date)
    print("End: " + end_date)
    print("Sentiment Score: " + str(sentiment_scores[i] + "/100\n"))
    print("Explanation: " + explanations[i] + "\n")
    print("Summary: " + summaries[i] + "\n")
    print("URLs and Summaries: ")
    for i, summary_and_url in enumerate(all_summaries_and_urls[i]):
        print("URL " + str(i+1) + ": " + summary_and_url[0])
        print("Summary: " + summary_and_url[1])
    print("________")
    

plt.plot(string_date_intervals, [int(score) for score in sentiment_scores])
plt.gcf().autofmt_xdate()
plt.xlabel("Date Intervals")
plt.ylabel('Sentiment Scores')
plt.ylim(0, 100)
plt.title(f"Sentiment Scores over time for {query.capitalize()}")
plt.show()