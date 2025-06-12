from http import HTTPStatus
import os
from pathlib import Path
from re import search
from tempfile import gettempdir
from typing import Optional
import json
from langchain_community.vectorstores import Chroma
import requests
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
import pandas as pd
from smolagents import DuckDuckGoSearchTool, WikipediaSearchTool, PythonInterpreterTool
from langchain_core.vectorstores import InMemoryVectorStore
from tabulate import tabulate
import re

load_dotenv(rf"C:\Users\Rushil\Desktop\training\Agentic AI\FinalProject\Final_Assignment_Template\env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
DATASET_PATH = rf"Agentic AI/gaia-benchmark-agent/dataset/metadata.jsonl"
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"



@tool
def multiply(a: int, b: int) -> int:
    """
    Add b to a.
    
    Args:
        a: first int number
        b: second int number
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return int(a / b)

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def wiki_search(query: str) -> str:
    """Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL.
    Args:
        query: The topic to search on Wikipedia."""
    search_tool = WikipediaSearchTool()
    return search_tool.forward(query=query)

@tool
def python_tool(code: str) -> str:
    """This is a tool that evaluates python code. It can be used to perform calculations.
    Args:
        code: The python code to run in interpreter."""
    pytool = PythonInterpreterTool()
    return pytool.forward(code=code)

@tool
def web_search(query: str) -> dict:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke(input=query)
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
         for doc in search_docs])
    return {"web_results": formatted_search_docs}

@tool
def arvix_search(query: str) -> dict:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
         for doc in search_docs])
    return {"arvix_results": formatted_search_docs}

@tool
def duck_duck_go_search(query: str) -> str:
    """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.
    Args:
        query: The search query.
    """
    search_tool = DuckDuckGoSearchTool()
    return search_tool.forward(query=query)

@tool
def excel_to_text(excel_path: str, sheet_name: str = "0") -> str:
    """Convert an Excel sheet into text.
    Args:
        excel_path: Path to the Excel file.
        sheet_name: Name or index of the Excel sheet.
    """
    path = Path(excel_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Excel file not found at {path}"
    sheet = int(sheet_name) if sheet_name.isdigit() else sheet_name
    try:
        data_frame = pd.read_excel(path, sheet_name=sheet)
        return data_frame.to_markdown(index=False) if hasattr(pd.DataFrame, "to_markdown") else \
            tabulate(data_frame, headers="keys", tablefmt="github", showindex=False)
    except Exception as exception:
        return f"Error reading Excel file: {exception}"


@tool 
def download_task_file(task_id: str, timeout: int = 60) -> Optional[str]:
    """
    Downloads a task file from the default API URL given a task_id.
    Args:
        task_id: The ID of the task to download the file for.
        timeout: The maximum time to wait for the download in seconds.
    Returns:
        The local path to the downloaded file, or None if an error occurred.
    """
    url = f"{DEFAULT_API_URL}/files/{task_id}"
    print(f"Downloading file for task_id: {task_id} from {url}")
    response = requests.get(url=url, timeout=timeout)
    if response.status_code == HTTPStatus.NOT_FOUND:
        print(f"File not found for task_id: {task_id}")
        return None
    try:
        response.raise_for_status()
        return save_task_file(task_id=task_id, response=response)
    except Exception as exception:
        print(f"Error downloading file for task_id {task_id}: {exception}")
        return None

def save_task_file(task_id: str, response: requests.Response) -> str:
    content_disposition = response.headers.get("content-disposition")
    match = search(r'filename="([^"]+)"', content_disposition) if content_disposition else None
    task_file_name = match.group(1) if match else task_id
    temp_path = Path(gettempdir()) / "task_files"
    temp_path.mkdir(exist_ok=True)
    task_file_path = temp_path / task_file_name
    with open(task_file_path, "wb") as file:
        file.write(response.content)
    print(f"File saved to: {task_file_path}")
    return str(task_file_path)

system_prompt = """
You are a helpful assistant tasked with answering questions using a set of tools. 
Now, I will ask you a question. Report your thoughts, and finish your answer with the following template: 
FINAL ANSWER: [YOUR FINAL ANSWER]. 
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
Your answer should only start with "FINAL ANSWER: ", then follows with the answer. 

Notes :

1. You should consider the answer as it is , do not alter anything.
2. If the values are comma seperated then consider it with commma(,) itself (e.g. "a,b").
"""

sys_msg = SystemMessage(content=system_prompt)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

with open(DATASET_PATH, 'r') as jsonl_file:
    json_list = list(jsonl_file)

json_QA = []
for json_str in json_list:
    json_data = json.loads(json_str)
    json_QA.append(json_data)

documents = []
for sample in json_QA:
    content = f"Question : {sample['Question']}\n\nFinal answer : {sample['Final answer']}"
    metadata = {"source": sample["task_id"]}
    documents.append(Document(page_content=content, metadata=metadata))

# Initialize vector store and add documents
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)
vector_store.persist()
print("Documents inserted:", vector_store._collection.count())


# Retriever tool (optional if you want to expose to agent)
retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    duck_duck_go_search,
    excel_to_text,
    python_tool,
    download_task_file 
]


def build_graph():
    """Build the graph using OpenAI"""
    print("Setting up OpenAI LLM")
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        print("Running assistant node")
        # Extract question and task_id from the latest HumanMessage
        latest_message_content = state["messages"][-1].content
        task_id_match = search(r"TASK_ID:\s*([^\n]+)", latest_message_content)
        question_match = search(r"QUESTION:\s*(.+)", latest_message_content, re.DOTALL)

        current_task_id = task_id_match.group(1).strip() if task_id_match else None
        current_question = question_match.group(1).strip() if question_match else latest_message_content 

       
        
        clean_messages = [msg for msg in state["messages"] if not isinstance(msg, HumanMessage) or not msg.content.startswith("TASK_ID:")]
        clean_messages.append(HumanMessage(content=current_question))
        return {"messages": [llm_with_tools.invoke(clean_messages)]} 

    def retriever(state: MessagesState):
        print(f"Running retriever node: {state}")
        latest_message_content = state["messages"][0].content 
        question_match = search(r"QUESTION:\s*(.+)", latest_message_content, re.DOTALL)
        question_for_retrieval = question_match.group(1).strip() if question_match else latest_message_content

        similar_question = vector_store.similarity_search(question_for_retrieval)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    print("Building graph")
    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()

# === Entry Point ===

if __name__ == "__main__":
    question = """¬(A ∧ B) ↔ (¬A ∨ ¬B)
¬(A ∨ B) ↔ (¬A ∧ ¬B)
(A → B) ↔ (¬B → ¬A)
(A → B) ↔ (¬A ∨ B)
(¬A → B) ↔ (A ∨ ¬B)
¬(A → B) ↔ (A ∧ ¬B)

Which of the above is not logically equivalent to the rest? Provide the full statement that doesn't fit."""
    

    graph = build_graph()
    
    messages = [HumanMessage(content=f"QUESTION: {question}")]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()