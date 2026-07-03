from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Search Tool
search_tool = TavilySearchResults(max_results=5)

# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Summarize the following news in simple bullet points.

{news}
"""
)

# Chain
chain = prompt | llm | StrOutputParser()

# Search latest AI news
news_result = search_tool.run("Latest AI news")

# Generate summary
result = chain.invoke(
    {
        "news": news_result
    }
)

print("\nAI News Summary:\n")
print(result)

# Tool Details
print("\nTool Information:")
print("Name:", search_tool.name)
print("Description:", search_tool.description)
print("Arguments:", search_tool.args)