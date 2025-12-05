from langchain.tools import tool

@tool
def get_local_news(city: str) -> str:
    """Get most recent local news"""
    return "The fly won the olimpic medal"


@tool
def get_user_location(user_greeting: str) -> str:
    """Retrieve user information based on user ID."""
    return "Brazil" if user_greeting == "Oi" else "USA"