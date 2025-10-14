import os
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor

# LangChain and document loading components
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pydantic import BaseModel, Field


class PDFInput(BaseModel):
    filepath: str = Field(description="The full path to the PDF file, e.g., 'papers/file.pdf'.")

# --- Helper Functions for PDF Processing ---
# These functions contain the core logic but are not exposed as tools to the agent.

def summarize_chunk_notes(llm: ChatOllama, chunk: str) -> List[str]:
    """
    Takes a single chunk of text and asks the LLM to summarize it into key bullet points.
    This is the "map" part of the map-reduce process.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You extract concise bullet points from a text. Output ONLY a JSON array of strings. Each string should be a key takeaway."),
            ("user",
             "Text chunk:\n---\n{chunk}\n---\nReturn a JSON array of the most important points.")
        ]
    )
    raw_response = (prompt | llm).invoke({"chunk": chunk}).content
    
    # Try to parse the JSON array from the LLM response
    try:
        start = raw_response.find("[")
        end = raw_response.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_response[start:end+1])
    except json.JSONDecodeError:
        # Fallback if JSON is malformed: split by lines
        return [line.strip("-•* ") for line in raw_response.splitlines() if line.strip()]
    return []

def reduce_notes(llm: ChatOllama, bullets: List[str]) -> List[str]:
    """
    Takes a large list of bullet points and asks the LLM to condense them.
    This is the "reduce" part of the map-reduce process.
    """
    joined_bullets = "\n".join(f"- {b}" for b in bullets)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a master synthesizer. Condense the following list of bullet points into a final, coherent list of the most critical takeaways. Output ONLY a JSON array of strings."),
            ("user",
             "Bullet points to condense:\n{notes}\n\nReturn the final, distilled list as a JSON array.")
        ]
    )
    raw_response = (prompt | llm).invoke({"notes": joined_bullets}).content

    # Try to parse the JSON array from the LLM response
    try:
        start = raw_response.find("[")
        end = raw_response.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_response[start:end+1])
    except json.JSONDecodeError:
        # Fallback: if condensing fails, return a truncated version of the original
        return bullets[:40]
    return []

def summarize_pdf_logic(filepath: str, llm: ChatOllama) -> str:
    """
    Orchestrates the PDF summarization, now requiring an LLM instance.
    """
    if not os.path.exists(filepath):
        return "Error: File not found at the specified path."

    print(f"\n--- Reading and processing {os.path.basename(filepath)} ---")
    
    reader = PdfReader(filepath)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not full_text.strip():
        return "Error: Could not extract text from the PDF."

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    all_bullets = []

    print(f"Summarizing {len(chunks)} chunks in parallel...")
    with ThreadPoolExecutor() as executor:
        # This now uses the llm object that was passed in
        results = executor.map(lambda chunk: summarize_chunk_notes(llm, chunk), chunks)
        for result in results:
            all_bullets.extend(result)
            
    print("Condensing notes into a final summary...")
    # This also uses the passed-in llm object
    final_bullets = reduce_notes(llm, all_bullets)
    
    return "\n".join(f"- {b}" for b in final_bullets)

# --- Agent Tools ---
# These are the functions the agent can actually see and choose to use.

@tool
def list_files_in_directory(directory: str = "papers") -> List[str]:
    """
    Lists all the file names in a specified directory.
    Use this first to see which PDF files are available locally.
    """
    # Clean the input to handle potential LLM formatting artifacts like quotes and newlines
    cleaned_directory = directory.strip().strip("'\"")
    
    print(f"\n--- (Listing files in '{cleaned_directory}') ---")
    try:
        return os.listdir(cleaned_directory)
    except FileNotFoundError:
        return ["Error: Directory not found."]

@tool
def read_and_summarize_pdf(filepath: str, model_name: str) -> str:
    """
    Reads a PDF file from a given path and returns a detailed summary.
    Only use this after confirming the file exists. You MUST provide the model_name.
    """
    try:
        cleaned_filepath = filepath.strip().strip("'\"")
        llm = ChatOllama(model=model_name, temperature=0.1, format="json")
        return summarize_pdf_logic(cleaned_filepath, llm)
    except Exception as e:
        return f"An unexpected error occurred while processing the PDF: {e}"