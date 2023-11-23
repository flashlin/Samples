import logging
import os
from urllib.parse import urlparse
import html2text
import markdown
import requests
from bs4 import BeautifulSoup
from io_utils import split_filename

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO)


class UrlsFile:
    urls = []

    def __init__(self, file: str):
        self.file = file
        if os.path.exists(file):
            with open(file, 'r') as file:
                for url in file:
                    self.urls.append(url.strip())

    def append(self, url: str):
        self.urls.append(url)
        with open(self.file, 'a') as file:
            file.write(url + '\r\n')

    def remove(self, url):
        new_urls = []
        for old_url in self.urls:
            if old_url == url:
                continue
            new_urls.append(old_url)
        self.urls = new_urls
        with open(self.file, 'w') as file:
            for url in new_urls:
                file.write(url + '\r\n')


def html_to_markdown(html_content: str):
    h = html2text.HTML2Text()
    h.body_width = 0
    markdown_text = h.handle(html_content)
    return markdown_text


def extract_filename_from_url(url: str):
    parsed_url = urlparse(url)
    filename = parsed_url.path.split("/")[-1]
    return filename


def get_target_filename(url: str):
    filename = extract_filename_from_url(url)
    file, ext = split_filename(filename)
    target_filename = f"./download/{file}.md"
    return target_filename


def save_html_to_file(url, html_content):
    target_file = get_target_filename(url)
    if target_file is None:
        return
    if target_file is not None and os.path.exists(target_file):
        return
    soup = BeautifulSoup(html_content, 'html.parser')
    # body_content = soup.body
    article_content = soup.find('div', id='articleContent')
    markdown_content = markdown.markdown(str(article_content), extensions=['markdown.extensions.fenced_code'])
    markdown_content = html_to_markdown(markdown_content)
    with open(target_file, 'w', encoding='utf-8') as file:
        file.write(markdown_content)


class Crawler:
    def __init__(self, urls: list[str]):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.visited_urls_file = UrlsFile('visited.txt')
        self.urls_to_visit_file = UrlsFile('urls_to_visit.txt')

    def download_url(self, url):
        return requests.get(url).text

    def get_linked_urls(self, url, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        all_a_tags = soup.find_all('a', class_='articleNode')
        for a_tag in all_a_tags:
            href = a_tag.get('href')
            yield href
        all_a_tags = soup.find_all('a', class_='node')
        for a_tag in all_a_tags:
            href = a_tag.get('href')
            yield href

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit_file.append(url)
            self.urls_to_visit.append(url)

    def crawl(self, url):
        if url in self.visited_urls:
            return
        html = self.download_url(url)
        self.visited_urls_file.append(url)
        save_html_to_file(url, html)
        for url in self.get_linked_urls(url, html):
            self.add_url_to_visit(url)

    def load_visited_urls(self):
        self.visited_urls = self.visited_urls_file.urls
        for url in self.urls_to_visit_file.urls:
            self.urls_to_visit.append(url)

    def run(self):
        self.load_visited_urls()
        while self.urls_to_visit:
            url = self.urls_to_visit.pop(0)
            self.urls_to_visit_file.remove(url)
            logging.info(f'Crawling: {url}')
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f'Failed to crawl: {url}')
            finally:
                self.visited_urls.append(url)
