import os 
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from transformers import AutoModel
import json
import torch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time 
from concurrent.futures import ThreadPoolExecutor
from PIL import Image 
from io import BytesIO
import undetected_chromedriver as uc
from google.colab import userdata 
load_dotenv()
pinecone = Pinecone(api_key= userdata.get('PINECONE_API_KEY')) 


def scrape_page(website):
    url = f'https://r.jina.ai/{website}'
    headers = {'Authorization': f"Bearer {userdata.get('JINA_API_KEY')}"} 

    response = requests.get(url, headers=headers)

    return response.text

def get_page_quality_relevant_chunks():
    index_name = 'summary-bot'
    index = pinecone.Index(index_name)
    namespace = 'US Rater Guidelines.pdf'
    query = 'E-E-A-T, YMYL, Lowest, Low, Medium, High, Highest, Page Quality'
    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    embeddings = embedding_model.encode(query).tolist()

    results = index.query(
        vector = embeddings,
        top_k = 10,
        namespace = namespace,
        include_metadata= True
    )
    formatted_results = [
        {
            'content': match['metadata']['document_content'],
            'metadata': match['metadata'],
            'relevance': float(match['score'])
        }
        for match in results['matches']
    ]
    
    json_output = {
        "query" : query,
        "results" : formatted_results
    }
    with open('Page Quality Guideline.json','w',encoding='utf-8') as file:
        json.dump(json_output,file)

def capture_desktop(url):
   #first capture screenshots and make sure selenium is doing job correctly on headless mode.
   # Then convert to PIL image and use that to send to LLM.
    chrome_options = Options()

# SSL and Security settings
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--disable-cookies')
    chrome_options.add_argument('--disable-notifications')

# WebGL and graphics settings
    chrome_options.add_argument('--disable-gpu-sandbox')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    #dynamically waits for page to load
    WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
    #creates bytes object 
    screenshot_1 = driver.get_screenshot_as_png()
    #BytesIO creates a virtual memory buffer to store the bytes, from there they are converted to a PIL object.
    desktop_start = Image.open(BytesIO(screenshot_1))
    #scroll halfway for another screenshot
    total_height = driver.execute_script("return document.body.scrollHeight")
    middle_height = total_height // 2
    driver.execute_script(f"window.scrollTo(0, {middle_height});")
    time.sleep(2)  # Allow time for any dynamic content to load
    
    screenshot_2 = driver.get_screenshot_as_png()
    desktop_mid = Image.open(BytesIO(screenshot_2))
    driver.quit()

    return desktop_start,desktop_mid

def capture_mobile(url):
    mobile_emulation = {
    "deviceMetrics": { "width": 360, "height": 640, "pixelRatio": 3.0 },
    "userAgent": "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36"
}
    chrome_options = Options()

# SSL and Security settings
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--disable-cookies')
    chrome_options.add_argument('--disable-notifications')

# WebGL and graphics settings
    chrome_options.add_argument('--disable-gpu-sandbox')
    chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
    screenshot_1 = driver.get_screenshot_as_png()
    mobile_start = Image.open(BytesIO(screenshot_1))
    #scroll halfway for another screenshot
    total_height = driver.execute_script("return document.body.scrollHeight")
    middle_height = total_height // 2
    driver.execute_script(f"window.scrollTo(0, {middle_height});")
    time.sleep(2)  # Allow time for any dynamic content to load

    screenshot_2 = driver.get_screenshot_as_png()
    mobile_end = Image.open(BytesIO(screenshot_2))
    driver.quit()

    return mobile_start, mobile_end
def capture_desktop_and_mobile_screenshots(url):
    #uses multithreading to save time
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        desktop_future = executor.submit(capture_desktop,url)
        mobile_future = executor.submit(capture_mobile,url)

        desktop_images = desktop_future.result()
        mobile_images = mobile_future.result()


    end_time = time.time()
    print(f"Screenshots captured in {end_time - start_time:.2f} seconds")
    return {
        'desktop':desktop_images,
        'mobile': mobile_images
    }



