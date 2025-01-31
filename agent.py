import google.generativeai as genai
import os 
import requests
from dotenv import load_dotenv
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from pinecone import Pinecone
from transformers import AutoModel
from langchain_community.embeddings import OpenAIEmbeddings
import json
import torch
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pinecone = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))
# embeddings = OpenAIEmbeddings(
#     openai_api_type=os.getenv('OPENAI_API_KEY'),
#      model = 'text-embedding-3-small',
#      dimensions = 768
#     )

def searching_unit(query):
    model = genai.GenerativeModel('models/gemini-1.5-pro-002')
    response = model.generate_content(contents=query,
                                tools='google_search_retrieval')
    return response.candidates[0].content.parts[0].text

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


def generate_guideline_relevant_queries(website):
    #first try with just 1 reasoning layer, plugging in scraped website text and research from google ground api
    #Then try plugging in relevant context from manual, using the first reasoning layer to come up with relevant queries and using those to extract relevant text. That relevant text would then be sent to 2nd reasoning layer alongside research text and website text.
    #OR use website research results to extract most relevant chunks from vector db
    #OR simply extract relevant chunks based on YMYL, EEAT, Lowest, Low, Medium, High, Fully Meets and use those chunks
    #However website experience still needs to be judged, a seprate evaluation on the websites design (need image model llm), ads, mobile experience, and content,
    research_prompt = f'Look into {website} trusworthiness and authority. Then give me a score for their EEAT (Experience,Expertise, Authiritativeness, Trust) as well and indication of whether they are YMYL (Your money your life) from clearly not YMYL, possible YMYL, likely YMYL, to clearly YMYL. Make sure to visit trustpilot as well when determining this.'
    # website_text = scrape_page(website)
    website_research = searching_unit(research_prompt)

    rating_prompt = f"""I'm going to give you text from a website as well as some research from it. From this you will give me a page quality rating according to US Rater guidelines (From lowest to fully meets)
            Website : {website}
            Website Research : {website_research}
            """
    
    #replace with deepseek R1 when available
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "system", "content": rating_prompt}
        ]
    )
    return completion.choices[0].message

# print(generate_guideline_relevant_queries('https://www.couponxoo.com/discount-nikon-scopes'))
get_page_quality_relevant_chunks()
# print(scrape_page('https://www.couponxoo.com/discount-nikon-scopes'))




    
        
            
    
    
