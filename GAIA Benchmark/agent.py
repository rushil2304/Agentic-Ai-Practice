from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilyExtract
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
import base64
import httpx
import os


load_dotenv(rf"C:\Users\Rushil\Desktop\training\Agentic AI\FinalProject\Final_Assignment_Template\env")
OPENAI_API_KEY =os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def add(a: int, b: int) -> int:
    """
    Add b to a.
    
    Args:
        a: first int number
        b: second int number
    """
    return a + b

@tool
def substract(a: int, b: int) -> int:
    """
    Subtract b from a.
    
    Args:
        a: first int number
        b: second int number
    """
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply a by b.
    
    Args:
        a: first int number
        b: second int number
    """
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """
    Divide a by b.
    
    Args:
        a: first int number
        b: second int number
    """
    if b == 0:
        raise ValueError("Can't divide by zero.")
    return a / b

@tool
def mod(a: int, b: int) -> int:
    """
    Remainder of a devided by b.
    
    Args:
        a: first int number
        b: second int number
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """
    Search Wikipedia.
    
    Args:
        query: what to search for
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "".join(
        [
            f'<START source="{doc.metadata["source"]}">{doc.page_content[:1000]}<END>'
            for doc in search_docs
        ])
    return {"wiki_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> str:
    """
    Search arXiv which is online archive of preprint and postprint manuscripts 
    for different fields of science.
    
    Args:
        query: what to search for
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "".join(
        [
            f'<START source="{doc.metadata["source"]}">{doc.page_content[:1000]}<END>'
            for doc in search_docs
        ])
    return {"arvix_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """
    Search WEB.
    
    Args:
        query: what to search for
    """
    search_docs = TavilySearchResults(max_results=3, include_answer=True).invoke({"query": query})
    formatted_search_docs = "".join(
        [
            f'<START source="{doc["url"]}">{doc["content"][:1000]}<END>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def open_web_page(url: str) -> str:
    """
    Open web page and get its content.
    
    Args:
        url: web page url in ""
    """
    search_docs = TavilyExtract().invoke({"urls": [url]})
    formatted_search_docs = f'<START source="{search_docs["results"][0]["url"]}">{search_docs["results"][0]["raw_content"][:1000]}<END>'
    return {"web_page_content": formatted_search_docs}

@tool
def youtube_transcript(url: str) -> str:
    """
    Get transcript of YouTube video.
    Args:
        url: YouTube video url in ""
    """    
    video_id = url.partition("https://www.youtube.com/watch?v=")[2]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([item["text"] for item in transcript])
    return {"youtube_transcript": transcript_text}


tools = [
    add,
    substract,
    multiply,
    divide,
    mod,
    wiki_search,
    arvix_search,
    web_search,
    open_web_page,
    youtube_transcript,
]

# System prompt
system_prompt = f"""
You are a general AI assistant. I will ask you a question.
First, provide a step-by-step explanation of your reasoning to arrive at the answer.
Then, respond with your final answer in a single line, formatted as follows: "FINAL ANSWER: [YOUR FINAL ANSWER]".
[YOUR FINAL ANSWER] should be a number, a string, or a comma-separated list of numbers and/or strings, depending on the question.
If the answer is a number, do not use commas or units (e.g., $, %) unless specified.
If the answer is a string, do not use articles or abbreviations (e.g., for cities), and write digits in plain text unless specified.
If the answer is a comma-separated list, apply the above rules for each element based on whether it is a number or a string.
"""
system_message = SystemMessage(content=system_prompt)

# Build graph
def build_graph():
    """Build LangGrapth graph of agent."""

    # Language model and tools
    llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        max_retries=2
    )
    llm_with_tools = llm.bind_tools(tools, strict=True)

    # Nodes
    def assistant(state: MessagesState):
        """Assistant node."""
        return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()


# Testing and solving particular tasks
if __name__ == "__main__":

    agent = build_graph()
    
    question = """
Review the chess position provided in the image. It is black's turn. 
Provide the correct next move for black which guarantees a win. 
Please provide your response in algebraic notation.
"""
    content_urls = {
        "image": "https://agents-course-unit4-scoring.hf.space/files/cca530fc-4052-43b2-b130-b30968d8aa44",
        "audio": None
    }
    
    # Define user message and add all the content
    content = [
        {
            "type": "text",
            "text": question
        }
    ]
    if content_urls["image"]:
        image_data = base64.b64encode(httpx.get(content_urls["image"]).content).decode("utf-8")
        content.append(
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/jpeg"
            }
        )
    if content_urls["audio"]:
        audio_data = base64.b64encode(httpx.get(content_urls["audio"]).content).decode("utf-8")
        content.append(
            {
                "type": "audio",
                "source_type": "base64",
                "data": audio_data,
                "mime_type": "audio/wav"
            }
        )
    messages = {
        "role": "user",
        "content": content
    }

    # Run agent on the question
    messages = agent.invoke({"messages": messages})
    for message in messages["messages"]:
        message.pretty_print()

    answer = messages["messages"][-1].content
    index = answer.find("FINAL ANSWER: ")
    
    print("\n")
    print("="*30)
    if index == -1:
        print(answer)
    print(answer[index+14:])
    print("="*30)