import google.generativeai as genai
import os 
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from transformers import AutoModel
from langchain_community.embeddings import OpenAIEmbeddings
import json
import torch
from PIL import Image
from lib.utlity import scrape_page,capture_desktop_and_mobile_screenshots
import torch
import time
# from google import genai


load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY')) # type: ignore
# genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
openai_client= OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # type: ignore


def research_evaluation(query):
    model = genai.GenerativeModel('models/gemini-1.5-pro-002')
    response = model.generate_content(contents=query,
                                tools='google_search_retrieval')
    return response.candidates[0].content.parts[0].text


def evaluate_page(website):
    #pass in screenshot on desktop, mobile, then send scraped page
    # scraped_page = scrape_page(website)
    mobile_desktop_screenshots = capture_desktop_and_mobile_screenshots(website)
    desktop_start, desktop_mid = mobile_desktop_screenshots['desktop']
    mobile_start, mobile_mid = mobile_desktop_screenshots['mobile']

    prompt = 'Evaluate these website screenshots and describe the user experience, design quality, and content organization. For the mobile screenshots note if the scaling or designing is off. ignore any cookie banners. '
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    response = model.generate_content([desktop_start,desktop_mid,mobile_start,mobile_mid,prompt])
   
    return response.candidates[0].content.parts[0].text

  

def page_quality_rating(website):
    start_time = time.time()
    research_prompt = f'Look into {website} trusworthiness and authority. Then give me a score for their EEAT (Experience,Expertise, Authiritativeness, Trust) as well and indication of whether they are YMYL (Your money your life) from clearly not YMYL, possible YMYL, likely YMYL, to clearly YMYL. Make sure to visit trustpilot as well when determining this. Be concise and make it around 5'
    # website_text = scrape_page(website)
    website_research = research_evaluation(research_prompt)
    page_evaluation = evaluate_page(website)
    with open('Page Quality Guideline.json','r') as file:
        relevant_chunks = json.load(file)
    rating_prompt = f"""I'm going to give you a website alongisde research conducted through searching. Then I will give you a visual evaluation based on screenshots. Then, I will give you relevant information from the US Rater guidelines for additional context.
         From this you will give me a page quality rating according to US Rater guidelines (From lowest to highest)
            Website : \n{website}\n
            Website Research : \n{website_research}\n
            'Page evaluation' : \n{page_evaluation}\n
            'US Rater Guidelines' : \n{relevant_chunks}\n

            Finally return in Json format like this:
            {{
                'Page Quality Rating' : 'Lowest to Highest'
            }}
            """
    print(rating_prompt)
    
    #replace with deepseek R1 when available
    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": rating_prompt}
        ],
    )
    end_time = time.time()
    print(f"Response completed in {end_time - start_time:.2f} seconds")
    return completion.choices[0].message.content

print(page_quality_rating('https://www.couponxoo.com/'))
# print(evaluate_page('https://www.couponxoo.com/'))


    
        
            
    
    
