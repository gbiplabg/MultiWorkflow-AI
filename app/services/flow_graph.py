from typing import Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()




llm= ChatGroq(model="openai/gpt-oss-120b",api_key=os.getenv('GROQ_API_KEY'))

class UserProfile(BaseModel):
    """Collected user information for LinkedIn post generation."""
    name: str
    email: str
    technology: str


collection_template = """
You are a helpful chatbot that guides a user step by step to create a LinkedIn post.

You must collect the following information:
1. Name
2. Valid Email address
3. Technology they are working on

Rules:
- Validate each input before moving to the next step.
- If the email is not valid, ask politely for correction.
- If the technology is vague or unknown, ask for clarification or suggest alternatives.
- Once you have all details, call the `UserProfile` tool with {name, email, technology}.
- Do not guess missing values. Always confirm with the user.
"""

llm_with_tool = llm.bind_tools([UserProfile])


class State(TypedDict):
    messages: Annotated[list, add_messages]


def info_chain(state: State):
    """Collect info step by step via LLM."""
    messages = [SystemMessage(content=collection_template)] + state["messages"]
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}


from langchain_core.messages import AIMessage

def linkedin_post_chain(state: State):
    """Generate LinkedIn post after collecting info."""
    # Find the latest AIMessage with a tool call
    tool_call_msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls),
        None,
    )
    if not tool_call_msg:
        raise ValueError("No tool call found in messages")

    tool_call = tool_call_msg.tool_calls[0]["args"]

    # LinkedIn Post Template
    post = (
        f"🚀 Exciting Update!\n\n"
        f"Hi everyone, I'm {tool_call['name']} and currently working with {tool_call['technology']}.\n"
        f"If you'd like to connect or collaborate, feel free to reach out at {tool_call['email']}.\n\n"
        f"#CareerGrowth #LinkedIn #Technology #Innovation"
    )

    return {"messages": [AIMessage(content=post)]}


def add_tool_message(state: State):
    """Add tool confirmation message after LLM tool call."""
    return {
        "messages": [
            ToolMessage(
                content="User profile collected successfully!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }

def get_state(state: State):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

memory = InMemorySaver()
workflow = StateGraph(State)

workflow.add_node("info", info_chain)
workflow.add_node("linkedin", linkedin_post_chain)
workflow.add_node("add_tool_message", add_tool_message)

workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "linkedin")
workflow.add_edge("linkedin", END)
workflow.add_edge(START, "info")

flow_graph = workflow.compile(checkpointer=memory)