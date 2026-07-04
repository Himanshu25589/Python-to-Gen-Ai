from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage
)
from rich import print


# Tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)


tools = {
    "get_text_length": get_text_length
}

# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Bind Tool
llm_with_tools = llm.bind_tools([get_text_length])

# Chat History
messages = []

# User Input
prompt = input("You: ")

messages.append(
    HumanMessage(content=prompt)
)

# First LLM Call
result = llm_with_tools.invoke(messages)

messages.append(result)

# Execute Tool if Needed
if result.tool_calls:

    tool_call = result.tool_calls[0]

    tool_name = tool_call["name"]

    tool_result = tools[tool_name].invoke(
        tool_call["args"]
    )

    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # Final LLM Call
    result = llm_with_tools.invoke(messages)

# Output
print("\n[bold green]AI:[/bold green]", result.content)