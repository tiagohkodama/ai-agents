# src/hello/main_graph.py

import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END

from .schemas import GreetingsResponse
from .tools import get_local_news, get_user_location


load_dotenv()


class GraphState(TypedDict, total=False):
    messages: List[Dict[str, Any]]
    location: Optional[str]
    news: Optional[str]
    structured_response: GreetingsResponse


def create_model():
    return init_chat_model(
        "gpt-4o-mini",
        temperature=0.5,
        timeout=10,
        base_url=os.environ.get("BASE_URL"),
        default_headers={"FlowAgent": "Flow Api"},
    )


SYSTEM_PROMPT = """
You are a friendly and funny person who always start chatting with most recent news.
You are an expert in making jokes.
You respond using local language.

You have access (via previous steps) to:
- user location (derived from greeting / language)
- local news (most recent)

You must ALWAYS return a structured response with the following fields:

- greeting: llm's hi
- news: local news. Example: The prime minister won yesterday.
- chat: llm starting chat. Use the news to start. Example: I heard that the minister won yesterday!

Always return the greeting, news and the chat initiation message.
"""


def node_get_location(state: GraphState) -> GraphState:
    messages = state.get("messages") or []
    if not messages:
        raise ValueError("No messages found in state")

    user_msg = messages[-1]["content"]
    location = get_user_location.run(user_msg)

    return {"location": location}


def node_get_news(state: GraphState) -> GraphState:
    """Usa a localização para buscar as notícias locais."""
    location = state.get("location") or "Unknown"
    news = get_local_news.run(location)
    return {"news": news}


def node_llm_response(state: GraphState) -> GraphState:
    model = create_model()
    structured_llm = model.with_structured_output(GreetingsResponse)

    location = state.get("location") or "Unknown"
    news = state.get("news") or "No news available."
    messages = state.get("messages") or []
    user_msg = messages[-1]["content"] if messages else ""

    # Passamos um system prompt equivalente ao usado no create_agent,
    # incluindo contexto de localização e notícia já resolvidos.
    chat_messages = [
        {
            "role": "system",
            "content": (
                SYSTEM_PROMPT
                + f"\nUser location: {location}\n"
                + f"Local news to use: {news}\n"
                + "Use these to fill greeting, news and chat fields."
            ),
        },
        {
            "role": "user",
            "content": user_msg,
        },
    ]

    structured_response: GreetingsResponse = structured_llm.invoke(chat_messages)

    return {"structured_response": structured_response}


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("get_location", node_get_location)
    graph.add_node("get_news", node_get_news)
    graph.add_node("llm_response", node_llm_response)

    graph.set_entry_point("get_location")
    graph.add_edge("get_location", "get_news")
    graph.add_edge("get_news", "llm_response")
    graph.add_edge("llm_response", END)

    return graph.compile()


def main():
    app = build_graph()

    initial_state: GraphState = {
        "messages": [{"role": "user", "content": "Oi"}],
    }

    final_state = app.invoke(initial_state)

    response = {"structured_response": final_state["structured_response"]}
    print(response["structured_response"])
    print("END")


if __name__ == "__main__":
    main()
