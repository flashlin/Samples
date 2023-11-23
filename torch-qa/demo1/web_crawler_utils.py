import logging
import requests
from bs4 import BeautifulSoup
from io_utils import split_filename
from web_crawler_lit import extract_filename_from_url, save_html_to_file, Crawler


def append_log(txt: str):
    with open('log.txt', 'a') as file:
        file.write(txt + '\r\n')

def crawl(url: str, target_file: str):
    response = requests.get(url)
    html_content = response.text
    save_html_to_file(url, html_content)


def crawl_all_article_urls(url: str):
    print("crawl article")
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    all_a_tags = soup.find_all('a', class_='articleNode')
    for a_tag in all_a_tags:
        href = a_tag.get('href')
        yield href


def crawl_all_node_urls(url: str):
    print("crawl nodes")
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    all_a_tags = soup.find_all('a', class_='node')
    for a_tag in all_a_tags:
        href = a_tag.get('href')
        yield href

def crawl_from_urls(urls):
    global all_urls
    for url in urls:
        if url in all_urls:
            continue
        all_urls[url] = url
        filename = extract_filename_from_url(url)
        file, ext = split_filename(filename)
        print(f"{url}")
        target_filename = f"./download/{file}.md"
        if file == '':
            crawl(url, None)
        else:
            crawl(url, target_filename)

all_urls = {}

def crawl_page(url: str):
    append_log(url)
    global all_urls
    crawling_urls = {}
    urls = crawl_all_node_urls(url)
    for url in urls:
        crawling_urls[url] = url
    urls = crawl_all_article_urls(url)
    for url in urls:
        crawling_urls[url] = url
    urls_to_crawl = []
    for url in list(crawling_urls.keys()):
        if url in all_urls:
            continue
        urls_to_crawl.append(url)
    crawl_from_urls(urls_to_crawl)


if __name__ == '__main__':
    # crawl_page("https://help.sbotop.com/article/35/100-deposit-bonus-terms-and-conditions-dep-1465.html")
    # crawl_page("https://help.sbotop.com/index.php?ln=en")
    Crawler(urls=['https://help.sbotop.com/index.php?ln=en']).run()

