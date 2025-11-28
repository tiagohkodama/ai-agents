import os
from dotenv import load_dotenv

# load .env
load_dotenv()

# LangChain agent + model imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# import our tools
from .tools import add, current_time
from .utils import pretty_print_agent_response

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your environment (.env)")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                     base_url=os.environ.get("BASE_URL"),
                     default_headers={"FlowAgent": "Flow Api"})

    agent = create_agent(
        model=llm,
        tools=[add, current_time],
        system_prompt="You are a helpful assistant that can call tools like 'add' and 'current_time'."
    )

    response = agent.invoke({"messages": [{"role": "user", "content": "What's 13 + 29? And what's the current time?"}]})

    pretty_print_agent_response(response)


if __name__ == "__main__":
    main()
