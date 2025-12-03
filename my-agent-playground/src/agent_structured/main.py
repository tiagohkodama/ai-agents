import os
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.agents.structured_output import ToolStrategy

class CityInfo(BaseModel):
    """City's geographic information"""
    name: str = Field(description="The name of the city")

class CitiesResponse(BaseModel):
    """Top cities information"""
    cities: List[CityInfo] = Field(description="Top cities list")

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
        response_format=ToolStrategy(CitiesResponse)
    )

    response = agent.invoke({ "messages": [ {"role": "user", "content": "Pegue a lista de cidades mais ricas do Brasil"} ] })
    
    print(response['structured_response'])
    print("END")

if __name__ == '__main__':
    main()