from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv

tavily = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))

@tool
def websearch(query : str)->str:
    """ Search the web for recent and reliable information on a topic.Return Titles,URLs, and snippets."""

    results = tavily.search(query = query,max_results = 5)

    return results

print(websearch.invoke("What are recent war news?"))
