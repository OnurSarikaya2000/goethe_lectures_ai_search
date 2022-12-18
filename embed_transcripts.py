import openai
import pinecone
from dotenv import load_dotenv
import os

# envs
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_KEY = os.environ["OPENAI_ORG_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]

# Pinecode setup
pinecone.init(api_key=PINECONE_API_KEY,
              environment="us-west1-gcp")

# open AI setup
openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY


txt_dir = 'vl_txt'

for transcript in os.listdir(txt_dir):
    transcript_file_name = os.fsdecode(transcript)

    response = openai.Embedding.create(
        input=os.path.join(
            txt_dir, transcript_file_name),
        model="text-embedding-ada-002"
    )
    embeds = [record['embedding'] for record in response['data']]

    index_name = 'goethe-linadi' + transcript_file_name
    # check if 'goethe-linadi' index already exists (only create index if not)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeds[0]))
    # connect to index
    index = pinecone.Index(index_name)
    break

    # res = index.query(
    #     vector=response['data'][0]['embedding'], top_k=5, include_values=True
    # )
    # for match in res['results'][0]['matches']:
    #     print(f"{match['score']:.2f}: {match['metadata']['text']}")
