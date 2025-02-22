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
import base64
from concurrent.futures import ThreadPoolExecutor
from PIL import Image 
from io import BytesIO
from playwright.async_api import async_playwright
# from undetected_playwright.async_api import async_playwright, Playwright

import asyncio

load_dotenv()
pinecone = Pinecone(api_key= os.getenv('PINECONE_API_KEY')) 


async def scrape_page(website):
    url = f'https://r.jina.ai/{website}'
    headers = {'Authorization': f"Bearer {os.getenv('JINA_API_KEY')}"} 

    response = requests.get(url, headers=headers)

def get_page_quality_relevant_chunks(query,file_name,k):
    index_name = 'summary-bot'
    index = pinecone.Index(index_name)
    namespace = 'US Rater Guidelines.pdf'
    # query = 'E-E-A-T, YMYL, Lowest, Low, Medium, High, Highest, Page Quality'
    embedding_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
    embeddings = embedding_model.encode(query).tolist()

    results = index.query(
        vector = embeddings,
        top_k =k,
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
    with open(file_name,'w',encoding='utf-8') as file:
        json.dump(json_output,file)

async def capture_desktop(url):
   #first capture screenshots and make sure selenium is doing job correctly on headless mode.
   # Then convert to PIL image and use that to send to LLM.

   async with async_playwright() as p:
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()
    await page.goto(url)
    #waits for page to load and dynamic content
    await page.wait_for_load_state('domcontentloaded') 

    screenshot_1 = await page.screenshot()
    desktop_start = Image.open(BytesIO(screenshot_1))
    #scroll halfway for another screenshot
    total_height =  await page.evaluate("document.body.scrollHeight")
    middle_height = total_height // 2
    await page.evaluate(f"window.scrollTo(0, {middle_height});")
    await page.wait_for_timeout(2000) # Allow time for any dynamic content to load
    
    screenshot_2 = await page.screenshot()
    # desktop_mid = base64.b64encode(screenshot_2).decode('utf-8')
    desktop_mid = Image.open(BytesIO(screenshot_2))
    await browser.close()

    return desktop_start,desktop_mid

async def capture_mobile(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 360, 'height': 640},
            user_agent='Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36'
        )
        page = await context.new_page()
        await page.goto(url)
        await page.wait_for_load_state('domcontentloaded')

        screenshot_1 = await page.screenshot()
        #convert to  base 64 string
        # mobile_start =base64.b64encode(screenshot_1).decode('utf-8')

        mobile_start = Image.open(BytesIO(screenshot_1))
        #scroll halfway for another screenshot
        total_height = await page.evaluate("document.body.scrollHeight")
        middle_height = total_height // 2
        await page.evaluate(f"window.scrollTo(0, {middle_height});")
        await page.wait_for_timeout(2000)  # Allow time for any dynamic content to load

        screenshot_2 = await page.screenshot()
        # mobile_end = base64.b64encode(screenshot_2).decode('utf-8')
        mobile_end = Image.open(BytesIO(screenshot_2))
        await browser.close()

        return mobile_start, mobile_end
async def capture_desktop_and_mobile_screenshots(url):
    #uses multithreading to save time
    start_time = time.time()
    #using asyncio to do multithreading
    desktop_images, mobile_images = await asyncio.gather(capture_desktop(url),capture_mobile(url))


    end_time = time.time()
    print(f"Screenshots captured in {end_time - start_time:.2f} seconds")
    return {
        'desktop':desktop_images,
        'mobile': mobile_images
    }

def get_screenshots(url):
    mobile_desktop_screenshots = capture_desktop_and_mobile_screenshots(url)
    desktop_start, desktop_mid = mobile_desktop_screenshots['desktop']
    mobile_start, mobile_mid = mobile_desktop_screenshots['mobile']
    encoded_images = [desktop_start,desktop_mid,mobile_start,mobile_mid]
    return encoded_images

# get_page_quality_relevant_chunks('Fails to meet,Slightly meets,Moderately meets,Highly meets,Fully meets, Query, User Intent','Page Quality Guidelines Needs met',5)




