import os
from dotenv import load_dotenv

# load .env
load_dotenv()

from langchain.chat_models import init_chat_model

from langchain.agents import create_agent
from .tools import Context, get_weather_for_location, get_user_location

from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

checkpointer = InMemorySaver()

SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    punny_response: str
    weather_conditions: str | None = None

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment (.env)")
    
    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.5,
        timeout=10,
        base_url=os.environ.get("BASE_URL"),
        default_headers={"FlowAgent": "Flow Api"})

    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[get_user_location, get_weather_for_location],
        context_schema=Context,
        response_format=ToolStrategy(ResponseFormat),
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": "1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
        config=config,
        context=Context(user_id="1")
    )

    print(response['structured_response'])

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )

    print(response['structured_response'])

    print(response)



if __name__ == "__main__":
    main()
