import requests
from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b

@tool
def divide(a: int, b: int) -> int:
    """Divides two numbers"""
    return a / b

@tool
def get_product(id: int):
    """Gets product details from API"""
    url = f"https://fakestoreapi.com/products/{id}"
    response = requests.get(url)
    data = response.json()
    return f"The product with id {id} is {data['title']} and costs {data['price']}"


tools = [add, multiply, divide, get_product]

tool_node = ToolNode(tools)

model = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

# final_state = app.invoke(
#     {"messages": [{"role": "user", "content": "What do we get if we multiply 4 with 12?"}]},
#     config={"configurable": {"thread_id": 42}}
# )

# print(final_state["messages"][-1].content)