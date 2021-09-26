from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import numpy as np


article_urls = []
targets = []
for difficulty in tqdm(np.arange(0,10,0.1)):
    URL = 'https://nplus1.ru/difficult/' + "{:2.1f}".format(difficulty)
    difficulty_page = requests.get(URL)
    difficulty_page_soup = BeautifulSoup(difficulty_page.text, "html.parser")
 
    for article_soup in difficulty_page_soup.findAll('article'):
        # article_soup = difficulty_page_soup.findAll('article')[0]
        article_url = 'https://nplus1.ru' + article_soup.find()['href']
        article_urls.append(article_url)
        targets.append(difficulty)
        
article_texts = []

for article_url in tqdm(article_urls):
    article_page = requests.get(article_url)
    article_page_soup = BeautifulSoup(article_page.text, "html.parser")
    div = article_page_soup.find('div', class_="body js-mediator-article")
    if div is not None:
        div_paragraphs = div.find_all('p', class_=None)
        article_text = ''
        for paragraph in div_paragraphs:
            paragraph_text = ''.join(paragraph.find_all(text=True,recursive=False))
            article_text += paragraph_text.replace(u'\xa0', u' ').replace(u'\n', u' ').replace(u'\r', u'')
        article_texts.append(article_text)
    else:
        article_texts.append('')

with open('article_texts.txt', 'w', encoding='utf-8') as f:
    for text in tqdm(list(article_texts)):
        f.write(text + '\n')
        
with open('targets.txt', 'w', encoding='utf-8') as f:
    for target in tqdm(targets):
        f.write("{:2.1f}".format(target) + '\n')