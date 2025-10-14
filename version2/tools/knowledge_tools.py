from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

@tool
def generate_summary_from_knowledge(topic: str, model_name: str) -> str:
    """
    Generates a detailed summary about a topic using the LLM's own internal knowledge.
    Use this tool as a last resort if you cannot find any relevant local PDF files.
    You MUST provide the model_name to use.
    """
    print(f"\n--- (Generating content for '{topic}' from internal knowledge using {model_name}) ---")
    
    llm = ChatOllama(model=model_name, temperature=0.4)
    
    summary_prompt = ChatPromptTemplate.from_template(
        "You are an expert on the topic of '{topic}'. "
        "Generate a detailed summary with key bullet points for a presentation."
    )
    
    summary_chain = summary_prompt | llm | StrOutputParser()
    return summary_chain.invoke({"topic": topic})