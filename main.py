import crawler.web_crawler as crawler
import pprint
import llm_api_client.open_ai_client as open_ai_clt
from dotenv import load_dotenv


load_dotenv()

query = input("Enter a query: ")
start_n_days_ago = int(input("Enter the number of days ago to start scraping from: "))
end_n_days_ago = int(input("Enter the number of days ago to end scraping from: "))

gs = crawler.GoogleScraper()
url_list = gs.scrape_query(query, start_n_days_ago, end_n_days_ago)
print("URLS TO BE SCRAPED:")
pprint.pprint(url_list)

us = crawler.URLScraper(url_list)
res = us.scrape_url_list()

ts = crawler.TextSummarizer()

summaries = []
for text in res:
    print("________")
    print("ORIGINAL:")
    print(text)
    summary = ts.summarize_text(text)
    summaries.append(summary)
    print("________")

for i, summary in enumerate(summaries):
    print("SUMMARY " + str(i+1) + ":")
    print(summary)
    print("________")
    
oa = open_ai_clt.OpenAIClient()

formatted_summaries = "\n\n".join(summaries)
sentiment_score, explanation, summary = oa.get_sentiment_and_summary(query, formatted_summaries)
print("\n\n________")
print("Sentiment Score: " + str(sentiment_score) + "/100")
print("\n\n")
print("Explanation: " + explanation)
print("\n\n")
print("Summary: " + summary)

#oa.question_davinci_model(f"I am going to give you a list of {len(formatted_summaries)} statements about {query}. Please rate each statement on a scale of 1-10, where 1 is very negative and 10 is very positive. These are the statements separated by newlines: {formatted_summaries}")