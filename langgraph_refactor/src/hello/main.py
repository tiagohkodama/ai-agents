import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents.structured_output import ToolStrategy
from .schemas import GreetingsResponse
from .tools import get_local_news, get_user_location

load_dotenv()

def main():
    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.5,
        timeout=10,
        base_url=os.environ.get("BASE_URL"),
        default_headers={"FlowAgent": "Flow Api"})

    agent = create_agent(
        model=model,
        system_prompt="""
            You are a friendly and funny person who always start chatting with most recent news.
            You are an expert in making jokes.
            You respond using local language
            You have access to two tools:

            - get_user_location: use this tool to figure out the location by language
            - get_local_news: use this tool to figure out the location news

            greeting: llm's hi
            news: local news. Example: The prime minister won yesterday.
            chat: llm starting chat. Use the news to start. Example: I heard that the minister won yesterday!

            Always return the greeting, news and the chat initiation message.
        """,
        tools=[get_user_location, get_local_news],
        response_format=ToolStrategy(GreetingsResponse)
    )

    response = agent.invoke({ "messages": [ {"role": "user", "content": "Oi"} ] })
    
    print(response['structured_response'])
    print("END")

if __name__ == '__main__':
    main()