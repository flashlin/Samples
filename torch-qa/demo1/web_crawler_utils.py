import requests
from bs4 import BeautifulSoup
import markdown
from urllib.parse import urlparse
import html2text
from io_utils import split_filename


def html_to_markdown(html_content: str):
    h = html2text.HTML2Text()
    h.body_width = 0
    markdown_text = h.handle(html_content)
    return markdown_text


def extract_filename_from_url(url: str):
    parsed_url = urlparse(url)
    filename = parsed_url.path.split("/")[-1]
    return filename


def crawl(url: str, target_file: str):
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')
    # body_content = soup.body
    article_content = soup.find('div', id='articleContent')
    markdown_content = markdown.markdown(str(article_content), extensions=['markdown.extensions.fenced_code'])
    markdown_content = html_to_markdown(markdown_content)
    with open(target_file, 'w', encoding='utf-8') as file:
        file.write(markdown_content)
    

def crawl_all_article_url(url: str):
    url = "https://help.sbotop.com/article/35/100-deposit-bonus-terms-and-conditions-dep-1465.html"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    all_a_tags = soup.find_all('a', class_='articleNode')
    for a_tag in all_a_tags:
        href = a_tag.get('href')
        yield href


if __name__ == '__main__':
    urls = crawl_all_article_url("https://help.sbotop.com/article/35/100-deposit-bonus-terms-and-conditions-dep-1465.html")
    for url in urls:
        filename = extract_filename_from_url(url)
        file, ext = split_filename(filename)
        print(f"{url}")
        target_filename = f"./download/{file}.md"
        crawl(url, target_filename)

