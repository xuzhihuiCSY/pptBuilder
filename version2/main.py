import sys
import requests
from typing import List
import os
from functools import partial
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import Tool

# Import the custom tools we created in the 'tools' directory
from tools.file_system_tools import list_files_in_directory, read_and_summarize_pdf
from tools.presentation_tools import create_powerpoint
from tools.knowledge_tools import generate_summary_from_knowledge

# --- Agent Setup ---
def get_local_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Gets the list of locally available Ollama models."""
    try:
        r = requests.get(f"{base_url}/api/tags")
        r.raise_for_status()
        # Return a list of model names, e.g., ['mistral:latest', 'llama3.1:8b']
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.RequestException:
        print("❌ Error: Could not connect to Ollama. Please ensure it is running.")
        return []

# 1. Set up the LLM
# This is the "brain" of our agent. Using a capable model is important for reasoning.
print("🔎 Checking for local Ollama models...")
local_models = get_local_ollama_models()
if not local_models:
    sys.exit(1) # Exit if no models are found or Ollama isn't running

# Automatically select the first available model
selected_model = local_models[0]
print(f"✅ Using model: {selected_model}")

llm = ChatOllama(model=selected_model, temperature=0.9)

# 2. List all the available tools for the agent
tools = [
    list_files_in_directory,
    read_and_summarize_pdf,
    generate_summary_from_knowledge,
    create_powerpoint,
]

# 3. Design the master prompt
# This is the most critical part. It's the set of instructions that guides the agent's thinking process.
prompt_template = f"""
You are a helpful research assistant who creates PowerPoint presentations.

**CRITICAL INSTRUCTION:** When you use a tool that requires an LLM (like `read_and_summarize_pdf`, `generate_summary_from_knowledge`, or `create_powerpoint`), you MUST include the `model_name` argument with the exact value: "{selected_model}"

**Your process is as follows:**
1.  Check for relevant local PDF files using `list_files_in_directory`.
2.  If a relevant file is found, summarize it using `read_and_summarize_pdf`.
3.  If no relevant file is found, generate content using `generate_summary_from_knowledge`.
4.  Once you have the content summary, create the presentation using the `create_powerpoint` tool.
5.  Finally, present the summary you used and confirm the presentation file has been created.
"""
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(prompt_template),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 4. Create the Agent and the Agent Executor
# The agent is the reasoning engine; the executor is what makes it run.
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Interactive Chat Loop ---

if __name__ == "__main__":
    print("🤖 PowerPoint Agent is ready. Let's create a presentation! (type 'exit' to quit)")
    
    # Check for Tavily API key
    if not os.environ.get("TAVILY_API_KEY"):
        print("\n⚠️  Warning: TAVILY_API_KEY environment variable not set.")
        print("   The agent will not be able to search the web.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Exiting. Goodbye!")
                break
            
            # Invoke the agent executor with the user's input
            result = agent_executor.invoke({"input": user_input})
            
            print(f"\n🤖 Agent: {result['output']}")

        except Exception as e:
            print(f"An error occurred: {e}")