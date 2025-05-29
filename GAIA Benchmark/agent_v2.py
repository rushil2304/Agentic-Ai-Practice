from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilyExtract
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
import base64
import httpx
import os
import tempfile

# Load environment variables (e.g., API keys) from .env file
load_dotenv(rf"C:\Users\Rushil\Desktop\training\Agentic AI\FinalProject\Final_Assignment_Template\env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
def divide(a: int, b: int) -> float:
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
    return {"wiki_results": "".join([f'<START source="{doc.metadata["source"]}">{doc.page_content[:1000]}<END>' for doc in search_docs])}

@tool
def arvix_search(query: str) -> str:
    """
    Search arXiv which is online archive of preprint and postprint manuscripts 
    for different fields of science.
    
    Args:
        query: what to search for
    """
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    return {"arvix_results": "".join([f'<START source="{doc.metadata["source"]}">{doc.page_content[:1000]}<END>' for doc in search_docs])}

@tool
def web_search(query: str) -> str:
    """
    Search WEB.
    
    Args:
        query: what to search for
    """
    search_docs = TavilySearchResults(max_results=3, include_answer=True).invoke({"query": query})
    return {"web_results": "".join([f'<START source="{doc["url"]}">{doc["content"][:1000]}<END>' for doc in search_docs])}

@tool
def open_web_page(url: str) -> str:
    """Fetches and extracts raw content from a given web URL."""
    search_docs = TavilyExtract().invoke({"urls": [url]})
    return {"web_page_content": f'<START source="{search_docs["results"][0]["url"]}">{search_docs["results"][0]["raw_content"][:1000]}<END>'}

@tool
def youtube_transcript(url: str) -> str:
    """Returns transcript of a YouTube video using its URL."""
    video_id = url.partition("https://www.youtube.com/watch?v=")[2]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return {"youtube_transcript": " ".join([item["text"] for item in transcript])}

tools = [add, substract, multiply, divide, mod, wiki_search, arvix_search, web_search, open_web_page, youtube_transcript]



system_prompt = """
You are a general AI assistant. I will ask you a question.
First, provide a step-by-step explanation of your reasoning to arrive at the answer.
Then, respond with your final answer in a single line, formatted as follows: "FINAL ANSWER: [YOUR FINAL ANSWER]".
[YOUR FINAL ANSWER] should be a number, a string, or a comma-separated list of numbers and/or strings, depending on the question.
If the answer is a number, do not use commas or units (e.g., $, %) unless specified.
If the answer is a string, do not use articles or abbreviations (e.g., for cities), and write digits in plain text unless specified.
If the answer is a comma-separated list, apply the above rules for each element based on whether it is a number or a string.
"""
system_message = SystemMessage(content=system_prompt)



def build_graph():
    """
    Builds a LangGraph agent with a base ChatOpenAI LLM, tools, and tool routing.
    """
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_retries=2)
    llm_with_tools = llm.bind_tools(tools, strict=True)

    def assistant(state: MessagesState):
        """Main assistant node that calls the LLM with the current state."""
        return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()



def transcribe_audio_from_url(url: str) -> str:
    """
    Downloads and transcribes an audio file from a URL using OpenAI Whisper API.
    
    Args:
        url (str): Publicly accessible URL of an audio file (.mp3, .wav, .ogg, etc.)
    
    Returns:
        str: Transcribed text from the audio.
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    suffix = os.path.splitext(url)[-1] or ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(httpx.get(url).content)
        tmp_path = tmp.name

    # Open the file separately so it's not locked by the previous context
    with open(tmp_path, "rb") as f:
        response = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers=headers,
            files={"file": (os.path.basename(tmp_path), f, "audio/mpeg")},
            data={"model": "whisper-1"}
        )

    os.remove(tmp_path)  
    return response.json().get("text", "")




if __name__ == "__main__":
    agent = build_graph()

    question = """
"give me conversion rate of dollars to euros for 100 dollars"
"""

    content_urls = {
        "image": None,
        "audio": None
    }
    
    # Prepare message content with optional audio/image
    content = [{"type": "text", "text": question}]

    if content_urls["image"]:
        content.append({"type": "image_url", "image_url": {"url": content_urls["image"]}})

    if content_urls["audio"]:
        transcript_text = transcribe_audio_from_url(content_urls["audio"])
        if transcript_text:
            content.insert(0, {"type": "text", "text": f"Transcribed audio: {transcript_text}"})

    messages = {
        "messages": [{"role": "user", "content": content}]
    }

    # Run the agent
    result = agent.invoke(messages)

    # Print results
    for message in result["messages"]:
        print(message.content)
