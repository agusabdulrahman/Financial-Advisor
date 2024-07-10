from typing import Dict, Optional, List

import hashlib
from pydantic import BaseModel
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window
from unstructured.staging.huggingface import stage_for_transformers
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.paths import DATA_DIR
from src.logger import get_console_logger

NEWS_FILE = DATA_DIR / 'news_2023-01-01_2023-01-05.json'
QDRANT_COLLECTION_NAME = 'alpaca_news'
QDRANT_VECTOR_SIZE = 384

logger = get_console_logger()

# tokenizer and LLM we use to embed the document text
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# init qdran client and collection where we store the news
from src.vector_db_api import get_qdrant_client, init_collection
qdran_client = get_qdrant_client()
qdran_client = init_collection(
    qdrant_client=qdran_client,
    collection_name=QDRANT_COLLECTION_NAME,
    vector_size=QDRANT_VECTOR_SIZE
)

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []
    
def parse_document(_data: Dict) -> Document:
    document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id = document_id)
    article_elements = partition_html(text =_data['content'])
    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements])))) 
    _data['headline'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['headline'])))
    _data['summary'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['summary'])))
    
    document.text = [_data['headline'], _data['summary'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['summary'] = _data['summary']
    # document.metadata['url'] = _data['url']
    # document.metadata['symbols'] = _data['symbols']
    # document.metadata['author'] = _data['author']
    document.metadata['date'] = _data['date']
    return document


def chunk(documen: Document) -> Document:
    
    
    