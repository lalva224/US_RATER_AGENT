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
from groq import Groq
# from google import genai


load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY')) # type: ignore
# genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
openai_client= OpenAI(api_key=os.getenv('OPENAI_API_KEY')) # type: ignore
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
with open('Page Quality Guideline.json','r') as file:
        page_quality_rating_context = json.load(file)
with open('Research Evaluation- EEAT and YMYL','r') as file:
        research_evaluation_context = json.load(file)


#TODO make use of logging 
def research_evaluation(website):
    research_prompt = f"""Look into {website} trusworthiness and authority. Then give me a score for their EEAT (Experience,Expertise, Authiritativeness, Trust) as well and indication of whether they are YMYL (Your money your life) from clearly not YMYL, possible YMYL, likely YMYL, to clearly YMYL.
      Make sure to visit trustpilot as well when determining this. Be very strict when it comes to trustworthiness. Be concise and give me around 5 sentences.
      """
    model = genai.GenerativeModel('models/gemini-1.5-pro-002')
    response = model.generate_content(contents=research_prompt,
                                tools='google_search_retrieval')
    return response.candidates[0].content.parts[0].text


def evaluate_page(website):
    #pass in screenshot on desktop, mobile, then send scraped page
    # scraped_page = scrape_page(website)
    mobile_desktop_screenshots = capture_desktop_and_mobile_screenshots(website)
    desktop_start, desktop_mid = mobile_desktop_screenshots['desktop']
    mobile_start, mobile_mid = mobile_desktop_screenshots['mobile']

    prompt = f""""Evaluate these website screenshots and describe the user experience, design quality, and content organization. 
    For the mobile screenshots note if the scaling or designing is off. ignore any cookie banners. Be concise and give me around 5 sentences
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    response = model.generate_content([desktop_start,desktop_mid,mobile_start,mobile_mid,prompt])
   
    return response.candidates[0].content.parts[0].text

  

def page_quality_rating(website):
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
            'US Rater Guidelines' : \n{page_quality_rating_context}\n

            Finally return in Json format like this:
            {{
                'Page Quality Rating' : '1-9'
            }}
            """
    print(rating_prompt)
    
  
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
    print(f"Response completed in {end_time - start_time:.2f} seconds")
    return response.choices[0].message.content


print(page_quality_rating('https://www.bostons.com/'))


    
        
            
    
    
