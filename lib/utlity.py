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
import threading
load_dotenv()
pinecone = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))

def scrape_page(website):
    url = f'https://r.jina.ai/{website}'
    headers = {'Authorization': f"Bearer {os.getenv('JINA_API_KEY')}"}

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
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    #could also try multithreading desktop and mobile screen capture to half the waiting time.
    # time.sleep(5)
    #dynamically waits for page to load
    WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
    driver.save_screenshot('Desktop Screen Capture.png')
    driver.quit()

def capture_mobile(url):
    mobile_emulation = {
    "deviceMetrics": { "width": 360, "height": 640, "pixelRatio": 3.0 },
    "userAgent": "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36"
}
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    WebDriverWait(driver, 20).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
    driver.save_screenshot('Mobile Screen Capture.png')

def capture_desktop_and_mobile_screenshots(url):
    #could take multiple screenshots as well, halfway through page.
    #uses multithreading to save time
    start_time = time.time()
    desktop_thread = threading.Thread(target=capture_desktop,args=(url,))
    mobile_thread = threading.Thread(target=capture_mobile,args=(url,))

    desktop_thread.start()
    mobile_thread.start()
    #wait to end, happens asynchronously
    desktop_thread.join()
    mobile_thread.join()

    end_time = time.time()
    print(f"Screenshots captured in {end_time - start_time:.2f} seconds")
capture_desktop_and_mobile_screenshots('https://www.bostons.com/')