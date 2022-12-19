from tqdm.auto import tqdm  # this is our progress bar
import openai
import pinecone
from dotenv import load_dotenv
import os
import itertools

# envs
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_KEY = os.environ["OPENAI_ORG_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
MODEL = "text-embedding-ada-002"  # use davinci when going prod

# Pinecode setup
pinecone.init(api_key=PINECONE_API_KEY,
              environment="us-west1-gcp")

# open AI setup
openai.organization = OPENAI_ORG_KEY
openai.api_key = OPENAI_API_KEY


txt_dir = 'vl_txt'
txt_content_array = []
txt_content = ''

for transcript in os.listdir(txt_dir):
    transcript_file_name = os.fsdecode(transcript)

    with open(os.path.join(txt_dir, transcript_file_name)) as f:
        txt_content_array.append(f.read())
        txt_content += f.read()

    txt_content += '. '
    break
print(txt_content_array)
res = openai.Embedding.create(
    input=txt_content_array, engine=MODEL
)
# extract embeddings to a list
embeds = [record['embedding'] for record in res['data']]

index_name = 'goethe-linadi'
# check if 'goethe-linadi' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=len(embeds))
index = pinecone.Index(index_name)

batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(txt_content), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(txt_content))
    # get batch of lines and IDs
    lines_batch = txt_content[i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))
