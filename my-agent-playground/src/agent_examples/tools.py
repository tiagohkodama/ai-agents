from langchain.tools import tool
from datetime import datetime
from typing import Optional

@tool
def add(a: int, b: int) -> int:
    """Add two integers. Usage: add(a: int, b: int) -> int"""
    return a + b

@tool
def current_time(timezone: Optional[str] = None) -> str:
    """Return current server time (timezone param is informative)."""
    now = datetime.utcnow()
    return f"UTC time now: {now.isoformat()}Z"
