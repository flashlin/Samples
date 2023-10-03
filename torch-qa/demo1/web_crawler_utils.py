import requests
from bs4 import BeautifulSoup
import markdown


def crawl(url: str, target_file: str):
    response = requests.get(url)
    html_content = response.text

    soup = BeautifulSoup(html_content, 'html.parser')
    # body_content = soup.body
    article_content = soup.find('a', class_='articleNode')
    markdown_content = markdown.markdown(str(article_content), extensions=['markdown.extensions.fenced_code'])
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
        print(f"{url=}")