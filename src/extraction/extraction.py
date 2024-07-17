import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.http import TextResponse
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import os

# Global dictionary to store URL and extracted text data.
url_text_dict = {}

class ConcordiaCrawler(scrapy.Spider):
    """
    A Scrapy Spider for crawling the Concordia University website.

    Attributes:
        name (str): Name of the spider.
        allowed_domains (list): List of allowed domains for scraping.
        start_urls (list): List of starting URLs for the spider.
        download_count (int): Counter for tracking number of downloads.
        custom_settings (dict): Custom settings for the spider.
    """

    name = 'concordia_crawler'
    allowed_domains = ['concordia.ca']
    start_urls = ['https://www.concordia.ca']
    download_count = 0   # Counter for tracking number of downloads.

    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
        'USER-AGENT': 'c_digi@live.concordia.ca'
    }

    def parse(self, response):
        """
        Parses the response from a web page.

        Args:
            response (TextResponse): The response object to process.

        Returns:
            None
        """
        if self.download_count >= self.download_limit:
            self.logger.info('Reached download limit, stopping spider.')
            self.crawler.engine.close_spider(self, 'download_limit_reached')
            return

        self.logger.info(f'Processing {response.url} with status {response.status}')
        if response.status == 200:
            content_type = response.headers.get('Content-Type', '').decode()
            if 'application/pdf' in content_type or 'video/mp4' in content_type:
                self.logger.info(f"Skipping non-text response at {response.url}")
                return

            if not (response.url in url_text_dict) and self.download_count < self.download_limit:
                soup = BeautifulSoup(response.text, 'html.parser')
                html_tag = soup.find('html')
                if html_tag and 'lang' in html_tag.attrs and html_tag.get('lang', '').startswith('en'):
                    # Focus on the main content of the page
                    main_content = soup.find('main')
                    if main_content:
                        # Remove script or style elements within the main content
                        for script_or_style in main_content.find_all(["script", "style"]):
                            script_or_style.extract()

                        # Get cleaned text from the main content
                        full_text = main_content.get_text(separator=' ', strip=True)
                        url_text_dict[response.url] = full_text
                        self.download_count += 1
                else:
                    self.logger.info(f"Skipping non-English or undefined language page at {response.url}")
                    return

            links = response.css('a::attr(href)').getall()
            for link in links:
                absolute_url = response.urljoin(link)
                if not absolute_url.startswith(("mailto:", "tel:")) and self.is_valid_url(absolute_url):  
                    if self.download_count < self.download_limit:
                        yield scrapy.Request(absolute_url, callback=self.parse)
                    else:
                        self.crawler.engine.close_spider(self, 'download_limit_reached')
                        return

    def is_valid_url(self, url):
        """
        Checks if a URL is valid and not of an ignored file type.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        if not (url.startswith("http://") or url.startswith("https://")):
            return False
        parsed_url = urlparse(url)
        _, ext = os.path.splitext(parsed_url.path)
        ignored_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.zip', '.xml', '.svg', '.mp3']
        return ext not in ignored_extensions

def write2File():
    """
    Writes extracted text data to files in a specified directory.
    """
    if not os.path.exists("collection"):
        os.makedirs("collection")

    for i, (url, text) in enumerate(url_text_dict.items()):
        file_name = f"./collection/doc{i}.txt"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(text)    

def extraction_pipeline():
    """
    Main function to start the crawling and writing process.
    """
    limit = int(input('Please enter a download limit: '))
    process = CrawlerProcess()
    process.crawl(ConcordiaCrawler, download_limit=limit)
    process.start()
    write2File()

def main():
    """
    Entry point of the script.
    """
    extraction_pipeline()

if __name__ == '__main__':
    main()
