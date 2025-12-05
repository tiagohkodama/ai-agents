from pydantic import BaseModel, Field

class GreetingsResponse(BaseModel):
    """The greeting response format"""
    greeting: str = Field(description="Respectful greeting")
    news: str = Field(description="The user local news")
    chat: str = Field(description="Machine way to chat")