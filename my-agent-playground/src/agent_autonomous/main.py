import os
import sqlite3
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

# =======================
#   Pydantic Models
# =======================

class Professor(BaseModel):
    name: str = Field(description="Full name of the teacher")
    available_days: List[str] = Field(description="Available days: mon, tue, wed, thu, fri")
    available_hours: List[int] = Field(description="Available hours represented as integers")
    subjects: List[str] = Field(description="List of subjects taught by the teacher")


class Subject(BaseModel):
    name: str = Field(description="Name of the subject")


class Slot(BaseModel):
    day: str = Field(description="Day of the week: mon, tue, wed, thu, fri")
    hour: int = Field(description="Class hour as an integer")
    subject: str = Field(description="Name of the subject")
    teacher: str = Field(description="Name of the teacher")
    classroom: str = Field(description="Class group name")


class Classroom(BaseModel):
    name: str = Field(description="Identifier name for the classroom group")
    subjects: List[Subject] = Field(description="List of subjects assigned to the class")
    slots: List[Slot] = Field(description="List of scheduled class slots")


class ScheduleResponse(BaseModel):
    classroom: Classroom = Field(description="Final structured schedule result")

# ==================================
#   SQLite Database Setup
# ==================================

def init_db():
    conn = sqlite3.connect("school.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS professor (
            name TEXT PRIMARY KEY
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS subject (
            name TEXT PRIMARY KEY
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS classroom (
            name TEXT PRIMARY KEY
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS slot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            day TEXT,
            hour INTEGER,
            subject TEXT,
            teacher TEXT,
            classroom TEXT
        )
    """)

    conn.commit()
    conn.close()


def persist_classroom(classroom: Classroom):
    conn = sqlite3.connect("school.db")
    c = conn.cursor()

    c.execute("INSERT OR IGNORE INTO classroom (name) VALUES (?)", (classroom.name,))

    for s in classroom.subjects:
        c.execute("INSERT OR IGNORE INTO subject (name) VALUES (?)", (s.name,))

    for slot in classroom.slots:
        c.execute("INSERT OR IGNORE INTO professor (name) VALUES (?)", (slot.teacher,))
        c.execute("""
            INSERT INTO slot (day, hour, subject, teacher, classroom)
            VALUES (?, ?, ?, ?, ?)
        """, (slot.day, slot.hour, slot.subject, slot.teacher, classroom.name))

    conn.commit()
    conn.close()

# ====================================
#      LLM Agent Setup
# ====================================

def create_schedule_agent():
    load_env()

    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.3,
        timeout=15,
        base_url=os.environ.get("BASE_URL"),
        default_headers={"FlowAgent": "Flow Api"}
    )

    return create_agent(
        model=model,
        response_format=ToolStrategy(ScheduleResponse)
    )


def load_env():
    load_dotenv()
    if not os.getenv("BASE_URL"):
        print("BASE_URL is missing in .env environment file.")

# ===============================
#      Main Logic
# ===============================

def main():
    init_db()
    agent = create_schedule_agent()

    user_input = """
    Teacher Leandro teaches Geography (one lesson per week), available on Monday before lunch.
    Teacher Tiago teaches Mathematics (3 lessons per week), available every day after 9 AM.
    Teacher Marcio teaches Chemistry (2 lessons per week), available at any time.
    The classroom group is 3A and needs: 1 Geography, 3 Mathematics and 2 Chemistry lessons.
    Available: Monday and Tuesday at 9, 10 and 11.
    """

    response = agent.invoke({
        "messages": [
            {
                "role": "system",
                "content": "You are a specialist in generating valid school timetables."
            },
            {
                "role": "user",
                "content": f"Generate a valid schedule based on this input: {user_input}. "
                           f"Respect teacher availability constraints. "
                           f"Return the response in the structure of the Classroom model."
            }
        ]
    })

    structured: Classroom = response["structured_response"].classroom

    persist_classroom(structured)

    print("Structured schedule saved successfully into school.db")
    print("Result:")
    print(structured)


if __name__ == "__main__":
    main()
