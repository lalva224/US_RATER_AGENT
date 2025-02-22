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
from groq import Groq
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from google import genai
from google.genai import types


#level 10 is debugging level 20 in INFO, then warning, error, and critical. It just means this level or higher. Then the formatting
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(message)s")


load_dotenv()

# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
google_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
# genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
with open('Page Quality Guideline.json','r') as file:
        page_quality_rating_context = json.load(file)
with open('Page Quality Guidelines Needs met.json','r') as file:
        needs_met_rating_context = json.load(file)


def needs_met_evaluation(scraped_page,query_image,query,user_location,user_intent):
      logging.info('Needs met evaluation started')
      img = Image.open(query_image)
      prompt = f"""Evaluate this  for needs met based on US rater Guidelines.
        Here is context from the guidelines : {needs_met_rating_context}
        It's worth noting that if this is a Special Content Result Block, you must only base your result on both the snippet and landing page evaluation.
        if this is a web search result block then only base result on the landing page evaluation.
        Here is scraped website : {scraped_page}
        Here is the query : {query}
        Here is user location : {user_location}
        Here is user intent : {user_intent}

        you will rate the needs met as either Fails to meet, Fails to meet+, Slightly meets, Slightly meets+, Moderately meets,Moderately meets+,Highly meets,Highly meets+, Fully meets. 
        The response will be in JSON and look like this:
        {{
        "needs met": ""
        }}

         """
      
      response = google_client.models.generate_content(model='gemini-2.0-flash-lite-preview-02-05',contents=[prompt,img])
      logging.info('needs met evaluation complete')
      return response.candidates[0].content.parts[0].text
def research_evaluation(website):
    logging.info('Starting Research Evaluation')
    research_prompt = f"""Look into {website} trusworthiness and authority. Then give me a score for their EEAT (Experience,Expertise, Authiritativeness, Trust) as well and indication of whether they are YMYL (Your money your life) from clearly not YMYL, possible YMYL, likely YMYL, to clearly YMYL.
      Make sure to visit trustpilot as well when determining this. Be very strict when it comes to trustworthiness. Be concise and give me around 5 sentences.
      """
    #flash lite does not flave searching available
    response = google_client.models.generate_content(
    model='gemini-2.0-flash',
    contents=research_prompt,
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            google_search=types.GoogleSearchRetrieval
        )]
    )
)
    logging.info('Research evaluation complete')
    return response.candidates[0].content.parts[0].text


async def evaluate_page(website):
    #pass in screenshot on desktop, mobile, then send scraped page
    logging.info('Starting page evaluation')
    mobile_desktop_screenshots = await capture_desktop_and_mobile_screenshots(website)
    desktop_start, desktop_mid = mobile_desktop_screenshots['desktop']
    mobile_start, mobile_mid = mobile_desktop_screenshots['mobile']

    scraped_page = scrape_page(website)
    prompt = f""""Evaluate these website screenshots and describe the user experience, design quality, and content organization. 
    For the mobile screenshots note if the scaling or designing is off. It is important to ignore any cookie banners. Be concise and give me around 5 sentences

    Here is the text from the website, note anything important : {scraped_page}


    """
    # model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite-preview-02-05")
    # response = model.generate_content([desktop_start,desktop_mid,mobile_start,mobile_mid,prompt])

    response = google_client.models.generate_content(model='gemini-2.0-flash-lite-preview-02-05',contents=[desktop_start,desktop_mid,mobile_start,mobile_mid,prompt])
    logging.info('Page evaluation complete')
    print(response.candidates[0].content.parts[0].text)
    return response.candidates[0].content.parts[0].text

  

def page_quality_rating(scraped_page,website):
    start_time = time.time()
    # website_text = scrape_page(website)
    website_research = research_evaluation(website)
    page_evaluation = evaluate_page(website)
    #TODO use website_research and page_evaluation to get the top_k results from the guideline, ideally k is 1 which is the correct category. Also consider providing manual context for every step, page evaluation, research evaluation, etc.
    rating_prompt = f"""I'm going to give you a website alongisde research conducted through searching. Then I will give you a visual evaluation based on screenshots. Then, I will give you relevant information from the US Rater guidelines for additional context.
         From this you will give me a page quality rating according to US Rater guidelines 
        Rate from Lowest, Lowest+, Low, Low+, Medium, Medium+, Migh, High+, Highest
            Website : \n{website}\n
            Website Research : \n{website_research}\n
            'Page evaluation' : \n{page_evaluation}\n
            'Here is the page content': \n {scraped_page}\n
            'US Rater Guidelines' : \n{page_quality_rating_context}\n


            Finally return in Json format like this:
            {{
                "Page Quality Rating" : "'
            }}
            """
    print(rating_prompt)
    
    logging.info('Starting final evaluation')
    response= client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": rating_prompt
        },
    ],
    model="deepseek-r1-distill-llama-70b",
    response_format={ "type": "json_object" }
)
    end_time = time.time()
    logging.info(f"Response completed in {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content

def get_page_ratings(website,query_image,query,user_location,user_intent):
    start_time = time.time()
    scraped_page = scrape_page(website)
    with ThreadPoolExecutor(max_workers=2) as executor:
         page_quality_rating_score = executor.submit(page_quality_rating,website,scraped_page)
         page_quality_rating_results = page_quality_rating_score.result()

         needs_met_rating_score = executor.submit(needs_met_evaluation,scraped_page,query_image,query,user_location,user_intent)
         needs_met_rating_results = needs_met_rating_score.result()
    
    end_time = time.time()
    logging.info(f"Response completed in {end_time - start_time:.2f} seconds")
    return page_quality_rating_results,needs_met_rating_results


url = 'https://www.nasa.gov/johnson/'
query_image = 'needs_met_rating_2.PNG'
query = 'nasa space center fl'
user_location = 'Not specified'
user_intent = 'Not specified'

print(get_page_ratings(url,query_image, query,user_location,user_intent))
# print(needs_met_evaluation(url,query_image,query,user_location,user_intent))
    
        
            
    
    
