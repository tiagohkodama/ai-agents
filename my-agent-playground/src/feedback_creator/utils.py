import os
from typing import List, Optional, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pathlib import Path

import csv


class FeedbackResponse(BaseModel):
    """LLM-generated feedback text for a person."""
    feedback: str = Field(
        description=(
            "Pequeno texto de agradecimento, em português do Brasil, "
            "com 2-4 frases, baseado na experiência positiva fornecida."
        )
    )


def create_feedback_agent():
    """Create and return a LangChain agent that outputs a FeedbackResponse."""
    load_dotenv()

    model = init_chat_model(
        "gpt-4o-mini",
        temperature=0.5,
        timeout=10,
        base_url=os.environ.get("BASE_URL"),
        default_headers={"FlowAgent": "Flow Api"},
    )

    agent = create_agent(
        model=model,
        response_format=ToolStrategy(FeedbackResponse),
    )
    return agent


def generate_feedback_csv(
    input_csv_path: str,
    output_csv_path: str,
    agent,
    name_column: str = "name",
    experience_column: str = "positive experience",
    feedback_column: str = "feedback",
):
    """
    Read input CSV, call LLM for each row, and write output CSV with feedback.

    - input_csv_path: path to the original CSV
    - output_csv_path: path where the completed CSV will be written
    - agent: LangChain agent created by create_feedback_agent()
    - name_column: column with the person's name
    - experience_column: column with the positive experience text
    - feedback_column: column where we will write the generated feedback
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)

    with open(input_csv_path, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])

        # Ensure feedback column exists
        if feedback_column not in fieldnames:
            fieldnames.append(feedback_column)

        rows_out: List[dict] = []

        for row in reader:
            name = (row.get(name_column) or "").strip()
            experience = (row.get(experience_column) or "").strip()

            # If there's no experience text, just leave feedback empty
            if not experience:
                row[feedback_column] = ""
                rows_out.append(row)
                continue

            prompt = f"""
Você é responsável por escrever pequenos textos de agradecimento para a equipe de uma escola infantil montessoriana.

A partir das informações abaixo, escreva um pequeno texto de feedback em nome da família do Keita.
para a pessoa mencionada, para ser entregue como um cartão de agradecimento.

Regras:
- Escreva em português do Brasil.
- Use 2 a 4 frases.
- Seja carinhoso e afetuoso, pois são pessoas muito queridas.
- A mensagem deve ser direcionado ao agradecimento e felicitações para as festas natalinas e ano novo.
- A mensagem deve conter agradecimentos se houver.
- Caso não tenha muitas informações deixe a mensagem um pouco mais genérica.
- Não invente fatos novos, apenas reformule e organize o que já está dito.
- Se o texto original estiver confuso ou com erros de digitação, corrija na sua resposta.
- Use o nome da pessoa no texto pelo menos uma vez.

Dados:
- Nome: {name}
- Experiência positiva: {experience}
"""

            try:
                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt.strip(),
                            }
                        ]
                    }
                )

                structured = cast(FeedbackResponse, response["structured_response"])
                feedback_text = structured.feedback
                print(feedback_text)

                row[feedback_column] = (feedback_text or "").strip()

            except Exception as e:
                row[feedback_column] = f"Erro ao gerar feedback: {e}"

            rows_out.append(row)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
