from googlesearch import search
import trafilatura
import datetime
import pprint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class WebCrawler:
    
    def __init__(self):
        self.scraper=GoogleScraper()
        self.url_scraper=URLScraper()
        self.summarizer=TextSummarizer()
        
    def crawl_query(self, query, num_pages, start_n_days_ago, end_n_days_ago):
        url_list = self.scraper.scrape_query(query, num_pages, start_n_days_ago, end_n_days_ago)
        print("URLS TO BE SCRAPED:")
        pprint.pprint(url_list)
        
        scraped_urls = self.url_scraper.scrape_url_list(url_list)
        if len(scraped_urls) == 0:
            print("No URLs were scraped")
            return []
        
        print("SCRAPED URLS:")
        pprint.pprint(scraped_urls)
        
        summaries = []
        for text in scraped_urls:
            print("________")
            print("ORIGINAL:")
            print(text)
            summary = self.summarizer.summarize_text(text)
            summaries.append(summary)
            print("________")
        
        for i, summary in enumerate(summaries):
            print("SUMMARY " + str(i+1) + ":")
            print(summary)
            print("________")
            
        return summaries


class GoogleScraper:
    
    FILTER_OUT_LIST_SITES = ["instagram.com/","youtube.com/", "spotify.com/", "music.apple.com/", "soundcloud.com/", ".gov/"]
    
    def __init__(self):
        return
    
    def scrape_query(self, query, num_pages_to_scrape, from_n_days_ago=0, to_n_days_ago=0):
        if not 0 <= to_n_days_ago <= from_n_days_ago:
            raise ValueError(f"from_n_days_ago: {from_n_days_ago} must be >= to_n_days_ago: {to_n_days_ago} must be >= 0")
        if not from_n_days_ago == to_n_days_ago == 0:
            curr_time = datetime.datetime.now()
            start_date, end_date = curr_time - datetime.timedelta(days=from_n_days_ago), curr_time - datetime.timedelta(days=to_n_days_ago)
            start_date, end_date = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

        print("SCRAPING FROM: " + start_date + " TO: " + end_date)
        full_query = query.strip(' ') + ' ' + ' '.join([f"-site:{site}" for site in self.FILTER_OUT_LIST_SITES]) + f" before:{start_date} after:{end_date}"
        url_list = search(full_query, num=10, stop=num_pages_to_scrape, pause=2)
        return list(url_list)


class URLScraper:
    
    LINE_REMOVAL_WORD_COUNT_THRESHOLD = 10
    
    def __init__(self):
        return
    
    def scrape_url_list(self, url_list):
        scraped_urls = []
        for url in url_list:
            print(f"Scraping {url}:\n")
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