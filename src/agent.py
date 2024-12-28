import os
from dotenv import load_dotenv
import sys
from datetime import datetime
from typing import TypedDict, Annotated, List, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from src.validators.agent_validators import *
from src.agent_tools import websearch_tool, retrieve_faq_info
from src.models import get_model

class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# def initialize_environment():
#     """Initialize environment variables and working directory"""
#     load_dotenv()
#     WORKDIR = os.getenv("WORKDIR")
#     if WORKDIR:
#         os.chdir(WORKDIR)
#         sys.path.append(WORKDIR)
#     return WORKDIR

def create_model():
    """Initialize and configure the model with tools"""
    tools = [websearch_tool, retrieve_faq_info]
    model = get_model('openai')
    return model.bind_tools(tools=tools), tools

def handle_tool_error(state) -> dict:
    """Handle errors from tool execution"""
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """Create a tool node with error handling"""
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], 
        exception_key="error"
    )

def should_continue(state: MessagesState) -> Literal["tools", "human_feedback"]:
    """Determine the next step in the workflow"""
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "human_feedback"

def should_continue_with_feedback(state: MessagesState) -> Literal["agent", "end"]:
    """Determine whether to continue with feedback"""
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, dict):
        if last_message.get("type","") == 'human':
            return "agent"
    if isinstance(last_message, HumanMessage):
        return "agent"
    return "end"

def create_system_message():
    """Create the system message with current timestamp"""
    return SystemMessage(content=f"""
        You are a helpful customer support assistant for HappyAI, a leading AI and big data analysis company.
        Use the provided tools to search for information about HappyAI's services, expertise, projects, and other company information to assist with user queries.
        When searching, be thorough and comprehensive. If initial queries don't yield sufficient results, try alternative phrasings or broader search terms.
        If a search returns no results, try reformulating your query before concluding the information isn't available.
        Always strive to provide accurate and up-to-date information about HappyAI's services and capabilities.
        \nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M, %A')}.
    """)

def call_model(state: MessagesState):
    """Process messages through the model"""
    messages = [create_system_message()] + state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def read_human_feedback(state: MessagesState):
    """Handle human feedback"""
    pass

def create_workflow():
    """Create and configure the workflow graph"""
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", create_tool_node_with_fallback(tools))
    workflow.add_node("human_feedback", read_human_feedback)
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "human_feedback": 'human_feedback',
            "tools": "tools"
        }
    )
    
    workflow.add_conditional_edges(
        "human_feedback",
        should_continue_with_feedback,
        {
            "agent": 'agent',
            "end": END
        }
    )
    
    workflow.add_edge("tools", 'agent')
    
    return workflow

# Initialize global components
# initialize_environment()
model, tools = create_model()

# Create and compile the workflow
workflow = create_workflow()
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=['human_feedback']
)

# if __name__ == '__main__':
#     # CLI interface for testing
#     while True:
#         question = input("Put your question: ")
#         for event in app.stream(
#             {"messages": [HumanMessage(content=question)]},
#             config={"configurable": {"thread_id": 42}}
#         ):
#             if event.get("agent","") == "":
#                 continue
#             else:
#                 msg = event['agent']['messages'][-1].content
#                 if msg == '':
#                     continue
#                 else:
#                     print(msg)
                    
# from IPython.display import Image, display

# try:
#     display(Image(app.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass
# from pathlib import Path

# # Get the Mermaid PNG bytes
# png_bytes = app.get_graph(xray=True).draw_mermaid_png()

# # Save to file
# output_path = Path("langchain_graph_mermaid.png")
# with open(output_path, "wb") as f:
#     f.write(png_bytes)

# print(f"Graph saved to: {output_path.absolute()}")
