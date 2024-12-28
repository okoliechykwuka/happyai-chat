import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.StreamHandler(),  # Output to console
        logging.FileHandler('pinecone_operations.log')  # Output to file
    ]
)

# Get the absolute path to the 'backend' directory
BACKEND_DIR = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(BACKEND_DIR)

from langchain_community.document_loaders import JSONLoader
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import Dict
from src.validators.pinecone_validators import IndexNameStructure, ExpectedNewData
from src.utils.env_setup import _set_env


# Load environment variables after setting up the path
load_dotenv()
# WORKDIR = os.getenv("WORKDIR")
# if WORKDIR:
#     os.chdir(WORKDIR)
#     logging.info(f"Changed working directory to: {WORKDIR}")

class PineconeManagment:
    def __init__(self):
        logging.info("Initializing PineconeManagement...")
        _set_env("OPENAI_API_KEY")
        _set_env("PINECONE_API_KEY")
        
        try:
            self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
            # logging.info("OpenAI Embeddings initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI Embeddings: {str(e)}")
            raise

        logging.info("PineconeManagement initialization completed")

    def __extract_metadata(self, record: dict, metadata: dict) -> dict:
        try:
            metadata["question"] = record['question']
            logging.debug(f"Extracted metadata for question: {record['question'][:50]}...")
            return metadata
        except KeyError as e:
            logging.error(f"Failed to extract metadata: Missing key {str(e)}")
            raise

    def reading_datasource(self):
        logging.info("Reading data source from JSON file...")
        try:
            loader = JSONLoader(
                file_path="data/faq.json",
                jq_schema='.[]',
                text_content=False,
                metadata_func=self.__extract_metadata
            )
            docs = loader.load()
            logging.info(f"Successfully loaded {len(docs)} documents from JSON")
            return docs
        except Exception as e:
            logging.error(f"Failed to read data source: {str(e)}")
            raise

    def creating_index(self, index_name: str, docs: Document, dimension=1536, metric="cosine"):
        logging.info(f"Attempting to create/verify index: {index_name}")
        try:
            IndexNameStructure(index_name=index_name)
            pc = Pinecone()
            existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
            
            if index_name in existing_indexes:
                logging.info(f"Index '{index_name}' already exists. Proceeding with existing index.")
                return
            
            logging.info(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name.lower(),
                dimension=dimension,    
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            while not pc.describe_index(index_name).status["ready"]:
                logging.info("Waiting for index to be ready...")
                time.sleep(1)

            logging.info(f"Index '{index_name}' created successfully")
            
            PineconeVectorStore.from_documents(
                documents=docs, 
                embedding=self.embedding, 
                index_name=index_name
            )
            logging.info(f"Index '{index_name}' populated with {len(docs)} documents")
            
        except Exception as e:
            logging.error(f"Error in creating_index: {str(e)}")
            raise

    def loading_vdb(self, index_name: str):
        logging.info(f"Loading vector database for index: {index_name}")
        try:
            self.vdb = PineconeVectorStore(
                index_name=index_name, 
                embedding=self.embedding
            )
            logging.info("Vector database loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load vector database: {str(e)}")
            raise

    def adding_documents(self, new_info: Dict[str, str]):
        logging.info("Adding new documents to vector database...")
        try:
            ExpectedNewData(new_info=new_info)
            doc = Document(
                page_content=f"question: {new_info['question']}\nanswer: {new_info['answer']}", 
                metadata={"question": new_info['question']}
            )
            self.vdb.add_documents([doc])
            logging.info("Documents added successfully")
        except Exception as e:
            logging.error(f"Failed to add documents: {str(e)}")
            raise

    def finding_similar_docs(self, user_query):
        logging.info(f"Searching for documents similar to query: {user_query[:50]}...")
        try:
            docs = self.vdb.similarity_search_with_relevance_scores(
                query=user_query,
                k=3,
                score_threshold=0.9
            )
            logging.info(f"Found {len(docs)} similar documents")
            return docs
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            raise