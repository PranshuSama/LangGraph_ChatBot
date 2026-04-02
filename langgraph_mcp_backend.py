from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import os
from pathlib import Path

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(model="llama-3.3-70b-versatile")

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=Z27VTFSKWQ2O1QEP"
    r = requests.get(url)
    return r.json()


mcp_servers = {}

# Local expense MCP server is preferred; remote endpoint is fallback.
expense_server_path = Path(
    os.getenv(
        "MCP_EXPENSE_SERVER_PATH",
        "/Users/pranshusama/Documents/MCP-Client/mcp-expense-server.py",
    )
)
if expense_server_path.exists():
    mcp_servers["expense"] = {
        "transport": "stdio",
        "command": "python3",
        "args": [str(expense_server_path)],
    }
else:
    mcp_servers["expense"] = {
        "transport": "streamable_http",  # if this fails, try "sse"
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
    }

# Local math MCP server is optional and can be configured via env var.
math_server_path = Path(
    os.getenv("MCP_MATH_SERVER_PATH", "/Users/pranshusama/Documents/MCP-Client/mcp-math-server.py")
)
if math_server_path.exists():
    mcp_servers["arith"] = {
        "transport": "stdio",
        "command": "python3",
        "args": [str(math_server_path)],
    }

client = MultiServerMCPClient(mcp_servers)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    try:
        response = await llm_with_tools.ainvoke(messages)
    except Exception as exc:
        error_text = str(exc)
        if "Failed to call a function" not in error_text and "failed_generation" not in error_text:
            raise
        response = await llm.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())