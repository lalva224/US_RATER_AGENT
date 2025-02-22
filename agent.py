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


async def needs_met_evaluation(website,query_image,query,user_location):
      logging.info('Needs met evaluation started')
      img = Image.open(query_image)
      #might need to get description of snippet through flash and then send to deepseek for reasoning.
      snippet_visual_prompt = f"""Break down what you see in the query snippet, use as much imagery as possible 
                query image : {query_image} ,
                query : {query}
                
      """
      snippet_visuals_response = google_client.models.generate_content(model='gemini-2.0-flash-lite-preview-02-05',contents=[snippet_visual_prompt,img])
      snippet_visuals = snippet_visuals_response.candidates[0].content.parts[0].text
      prompt = f"""
         Evaluate this result for needs met based on US rater Guidelines.
        Here is context from the guidelines : 
        Fails to Meet (FailsM): This result completely fails to address the user’s needs. It may be off-topic, contain incorrect or outdated information, or address a very unlikely interpretation of the query. It can also include harmful, misleading, or offensive content.

    Slightly Meets (SM): This result is less helpful for a reasonable query interpretation or user intent. It may be relevant but not fully address the core need, or it might be helpful for an unlikely interpretation. Sometimes, it may provide outdated or incomplete information.

    Moderately Meets (MM): This result fits the query but is less satisfying than highly relevant results. It might contain slightly harder-to-understand content or outdated information, or may be less representative of real-world opinions. It is still useful but not as valuable.

    Highly Meets (HM): A very helpful result for any reasonable query interpretation. It directly addresses the core user intent with high relevance and satisfies most users.

    Fully Meets (FM): This result fully satisfies the user's needs and provides all the information the user is looking for with complete accuracy and relevance.
            also keep in mind Only 3 types of questions can receive a fully meets rating: 
    Know simple (specific fact), website query, device action query (seeking to complete a specific task or action on a device)

        Remember that the needs met is based on how accuarely the result meets the user's query, for instance if they searched for nasa center fl but receive nasa center houston that is fails to meet.
        Here is scraped website : {scraped_page}
        Here is the query : {query}
        Here is user location : {user_location}

        Here are the snippet visuals, only take this into account if it is a special result block : {snippet_visuals}
        you will rate the needs met as either Fails to meet, Fails to meet+, Slightly meets, Slightly meets+, Moderately meets,Moderately meets+,Highly meets,Highly meets+, Fully meets. 
        The response will be in JSON and look like this:


        Pay very close attention to the content of the scraped page in determing user intent, think carefully and step by step
        Keep in mind the in between ratings with the +
        {{
        "needs met": ""
        }}

         """
      response= client.chat.completions.create(
      messages=[
            {
                "role": "system",
                "content": prompt
            },
        ],
        model="deepseek-r1-distill-llama-70b",
        response_format={ "type": "json_object" }
    )
      logging.info('needs met evaluation complete')
      return response.choices[0].message.content
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


async def evaluate_page(website,scraped_page):
    #pass in screenshot on desktop, mobile, then send scraped page
    logging.info('Starting page evaluation')
    mobile_desktop_screenshots = await capture_desktop_and_mobile_screenshots(website)
    desktop_start, desktop_mid = mobile_desktop_screenshots['desktop']
    mobile_start, mobile_mid = mobile_desktop_screenshots['mobile']

    prompt = f""""Evaluate these website screenshots and describe the user experience, design quality, and content organization. 
    For the mobile screenshots note if the scaling or designing is off. It is important to ignore any cookie banners. Be concise and give me around 5 sentences

    Here is the text from the website, note anything important : {scraped_page}


    """

    response = google_client.models.generate_content(model='gemini-2.0-flash-lite-preview-02-05',contents=[desktop_start,desktop_mid,mobile_start,mobile_mid,prompt])
    logging.info('Page evaluation complete')
    
    return response.candidates[0].content.parts[0].text

  

async def page_quality_rating(website,scraped_page):
    start_time = time.time()
    website_research = research_evaluation(website)
    page_evaluation = await evaluate_page(website,scraped_page)
    #TODO use website_research and page_evaluation to get the top_k results from the guideline, ideally k is 1 which is the correct category. Also consider providing manual context for every step, page evaluation, research evaluation, etc.
    rating_prompt = f"""I'm going to give you a website alongside research conducted through searching. Then I will give you a visual evaluation based on screenshots. Then, I will give you relevant information from the US Rater guidelines for additional context.
         From this you will give me a page quality rating according to US Rater guidelines 
        Rate from Lowest, Lowest+, Low, Low+, Medium, Medium+, Migh, High+, Highest
            Website : \n{website}\n
            Website Research : \n{website_research}\n
            'Page evaluation' : \n{page_evaluation}\n
            'US Rater Guidelines' : \n{page_quality_rating_context}\n
            As a summary you will keep track of the purpose of the page and how well it accomplishes it, EEAT, and then you will grade more harshly if you categorize this as YMYL
            If you note the page harmful,untrustworthy, or spammy then it must be rated lowest. 
            If the page has disruptive design then low must be given
            Keep in mind the in between ratings with the +

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

async def get_page_ratings(website,query_image,query,user_location):
    start_time = time.time()
    scraped_page = await scrape_page(website)
    page_quality_results,needs_met_rating_results = await asyncio.gather(
         page_quality_rating(website,scraped_page),
         needs_met_evaluation(website,query_image,query,user_location)
    )
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #      page_quality_rating_score = executor.submit(page_quality_rating,website,scraped_page)
    #      page_quality_rating_results = page_quality_rating_score.result()

    #      needs_met_rating_score = executor.submit(needs_met_evaluation,website,scraped_page,query_image,query,user_location)
    #      needs_met_rating_results = needs_met_rating_score.result()
    
    end_time = time.time()
    logging.info(f"Response completed in {end_time - start_time:.2f} seconds")
    return page_quality_results,needs_met_rating_results


url = 'https://adfs.geoworkerz.com/adfs/ls?version=1.0&action=signin&realm=urn%3AAppProxy%3Acom&appRealm=cc17932c-5c3d-ec11-a97b-000d3a1de952&returnUrl=https%3A%2F%2Fgeoworkerz.telusinternational.ai%2FCommunity%2520Portal%2FQuality%2FSiteAssets%2FImages%2FRecognizing%2520Low%2520Quality%2520Pages%2520Quiz%2520Images%2Fdisruptive_page_design.JPG&client-request-id=81D2C67C-84DF-0000-FDF8-D481DF84DB01'
query_image = 'needs_met_rating_7.png'
query = 'praying kesha song chords'
user_location = 'Not specified'

async def main():
    scraped_page = await scrape_page(url)
    page_ratings = await page_quality_rating(url,scraped_page)
    print(page_ratings)

if __name__ =='__main__':
     asyncio.run(main())
