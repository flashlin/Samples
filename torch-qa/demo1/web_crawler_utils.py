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
    
