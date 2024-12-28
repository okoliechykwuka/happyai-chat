import os
from pathlib import Path
from dotenv import load_dotenv
import sys


# load_dotenv()
# WORKDIR=os.getenv("WORKDIR")
# os.chdir(WORKDIR)
# sys.path.append(WORKDIR)

from langchain_core.tools import tool
from typing import  Literal
import pandas as pd
import json
from langchain_community.tools.tavily_search import TavilySearchResults

from src.validators.agent_validators import *
from src.vector_database.vector_db import PineconeManagment
from src.models import format_retrieved_docs



pinecone_conn = PineconeManagment()
pinecone_conn.loading_vdb(index_name = 'happyai')

websearch_tool = TavilySearchResults(max_results=2)


@tool
def retrieve_faq_info(question:str):
    """
    Retrieves information about the HappyAI platform from relevant documents.
    
    Use this tool to answer queries about HappyAI, such as:
    - "What are HappyAI's main services?"
    - "How long has HappyAI been operating?"
    - "What is HappyAI's expertise?"
    
    Parameters:
        question (str): The query about HappyAI to look up
        
    Returns:
        Information retrieved from the RAG chain based on the query
    """
    retriever = pinecone_conn.vdb.as_retriever(search_type="similarity", 
                                    search_kwargs={"k": 1})
    rag_chain = retriever | format_retrieved_docs
    return rag_chain.invoke(question)




