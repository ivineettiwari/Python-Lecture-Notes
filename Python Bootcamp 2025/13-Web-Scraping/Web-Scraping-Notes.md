# Web Scraping Notes

---

## 1. Overview of Web Scraping in Python

Web scraping is the process of automatically extracting data from websites. It is widely used for data collection, research, price monitoring, news aggregation, and more. Python is a popular language for web scraping due to its rich ecosystem of libraries and ease of use.

---

## 2. Common Tools and Libraries

- **requests**: For sending HTTP requests to web pages.
- **BeautifulSoup**: For parsing and navigating HTML/XML documents.
- **lxml**: Fast XML and HTML parser.
- **Scrapy**: A powerful and scalable web scraping framework.
- **Selenium**: For automating browsers, useful for scraping dynamic content.
- **pandas**: For data manipulation and storage.

---

## 3. Best Practices

- **Respect robots.txt**: Always check the website’s robots.txt file to see what is allowed to be scraped.
- **Rate Limiting**: Avoid sending too many requests in a short time. Use time.sleep() to pause between requests.
- **User-Agent**: Set a user-agent header to mimic a real browser.
- **Error Handling**: Handle exceptions and failed requests gracefully.
- **Data Storage**: Store scraped data in structured formats (CSV, JSON, databases).
- **Avoiding Bans**: Rotate IPs and user-agents if scraping at scale.

---

## 4. Legal Considerations

- **Terms of Service**: Always review the website’s terms of service before scraping.
- **Copyright**: Do not scrape copyrighted material for redistribution.
- **Personal Data**: Be cautious when scraping personal or sensitive data.
- **API Availability**: Prefer using official APIs if available.

---

## 5. Code Examples

### 5.1 Using requests and BeautifulSoup
```python
import requests
from bs4 import BeautifulSoup

url = 'https://example.com'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Example: Extract all links
for link in soup.find_all('a'):
    print(link.get('href'))
```

### 5.2 Using Scrapy
- Install Scrapy: `pip install scrapy`
- Create a new Scrapy project: `scrapy startproject myproject`
- Example spider:
```python
import scrapy

class ExampleSpider(scrapy.Spider):
    name = 'example'
    start_urls = ['https://example.com']

    def parse(self, response):
        for link in response.css('a::attr(href)').getall():
            yield {'link': link}
```

### 5.3 Using Selenium (for dynamic content)
```python
from selenium import webdriver
from selenium.webdriver.common.by import By

# Make sure to download the appropriate WebDriver for your browser
browser = webdriver.Chrome()
browser.get('https://example.com')

# Example: Extract all links
links = browser.find_elements(By.TAG_NAME, 'a')
for link in links:
    print(link.get_attribute('href'))
browser.quit()
```

### 5.4 Downloading and Saving Files

#### Downloading and Saving an Image
```python
import requests

img_url = 'https://www.example.com/image.jpg'
response = requests.get(img_url)
with open('image.jpg', 'wb') as f:
    f.write(response.content)
```

#### Downloading and Saving a PDF
```python
import requests

pdf_url = 'https://www.example.com/sample.pdf'
response = requests.get(pdf_url)
with open('sample.pdf', 'wb') as f:
    f.write(response.content)
```

#### Saving Scraped Data to CSV
```python
import csv

data = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25}
]

with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['name', 'age'])
    writer.writeheader()
    writer.writerows(data)
```

#### Downloading Multiple Files from a Webpage
```python
import requests
from bs4 import BeautifulSoup
import os

url = 'https://www.example.com/gallery'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

os.makedirs('images', exist_ok=True)
for img in soup.find_all('img'):
    img_url = img['src']
    img_name = os.path.basename(img_url)
    img_data = requests.get(img_url).content
    with open(f'images/{img_name}', 'wb') as handler:
        handler.write(img_data)
```

---

## 6. Summary of Notebooks in This Folder

- **00-Guide-to-Web-Scraping.ipynb**: Introduction and overview of web scraping concepts, tools, and ethics.
- **01-Web-Scraping-Exercises.ipynb**: Practice exercises for scraping static and dynamic web pages.
- **02-Web-Scraping-Exercise-Solutions.ipynb**: Solutions and explanations for the exercises.

---

## 7. Personal Notes Template

### Website/Project:
- URL:
- Data to Extract:
- Tools/Libraries Used:
- Challenges Faced:
- Solutions/Workarounds:
- Date:

---

Feel free to expand these notes with your own findings, code snippets, and tips as you progress in your web scraping journey!
