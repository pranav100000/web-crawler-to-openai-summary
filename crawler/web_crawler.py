from googlesearch import search
from bs4 import BeautifulSoup
import trafilatura
from trafilatura import sitemaps
import requests
import datetime
import pprint
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ScrapedWebPage:
    pass

class GoogleScraper:
    
    FILTER_OUT_LIST_SITES = ["instagram.com/","youtube.com/", "spotify.com/", "music.apple.com/", "soundcloud.com/", ".gov/"]
    FILTER_OUT_TERMS = ["music", "video"]
    
    def __init__(self):
        return
    
    def scrape_query(self, query, from_n_days_ago=0, to_n_days_ago=0):
        if not 0 <= to_n_days_ago <= from_n_days_ago:
            raise ValueError("from_n_days_ago and to_n_days_ago must be >= 0")
        time_param = None
        if not from_n_days_ago == to_n_days_ago == 0:
            curr_time = datetime.datetime.now()
            start_date, end_date = curr_time - datetime.timedelta(days=from_n_days_ago), curr_time - datetime.timedelta(days=to_n_days_ago)
            start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            time_param = f"cdr:1,cd_min:{start_date},cd_max:{end_date}"

        print("SCRAPING FROM: " + start_date + " TO: " + end_date)
        print("TIME PARAM: " + str(time_param))
        
        print("Google Query: " + query)
        
        def filter_scraped_urls(url):
            return not any([x in url for x in self.FILTER_OUT_LIST])
        
        full_query = query.strip(' ') + ' ' + ' '.join([f"-site:{site}" for site in self.FILTER_OUT_LIST_SITES]) + f" before:{start_date} after:{end_date}"
        print(full_query)
        url_list = search(full_query, num=10, stop=10, pause=2)
        
        #url_list = filter(filter_scraped_urls, url_list)
        
        #print(list(url_list))
        
                
        return list(url_list)


class URLScraper:
    
    LINE_REMOVAL_WORD_COUNT_THRESHOLD = 10
    
    def __init__(self, url_list, base_url=None):
        self.url_list = url_list
        self.base_url = base_url
        return
    
    def scrape_url_list(self):
        scraped_urls = []
        for url in self.url_list:
            print(f"___________\nScraping {url}:\n___________")
            downloaded = trafilatura.fetch_url(url)
            result = trafilatura.extract(downloaded, target_language="EN", include_tables=False, include_comments=False, favor_precision=True, deduplicate=True, only_with_metadata=True)
            if result is None:
                continue
            line_list = result.split("\n")
            removed_short_lines_list = [line for line in line_list if len(line) > self.LINE_REMOVAL_WORD_COUNT_THRESHOLD]
            scraped_urls.append('\n'.join(removed_short_lines_list))
        return(scraped_urls)


class TextSummarizer:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained('T5-base')
        self.model=AutoModelForSeq2SeqLM.from_pretrained('T5-base', return_dict=True)
    
    def summarize_text(self, text):
        if text is None:
            return "N/A"
        inputs=self.tokenizer.encode("summarize: " +text,return_tensors='pt', max_length=512, truncation=True)
        output = self.model.generate(inputs, min_length=80, max_length=100)
        summary=self.tokenizer.decode(output[0], skip_special_tokens=True)
        return summary


if __name__ == "__main__":

    gs = GoogleScraper()
    url_list = gs.scrape_query("gucci mane ", 100, 90)
    print("URL LIST:")
    pprint.pprint(url_list)

    us = URLScraper(url_list)
    res = us.scrape_url_list()

    ts = TextSummarizer()

    summaries = []
    for text in res:
        print("________")
        print("ORIGINAL:")
        print(text)
        print("SUMMARY:")
        summary = ts.summarize_text(text)
        summaries.append(summary)
        print(summary)
        print("________")

    for i, summary in enumerate(summaries):
        print("SUMMARY " + str(i+1) + ":")
        print(summary)
        print("________")



