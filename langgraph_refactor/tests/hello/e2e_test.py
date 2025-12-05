# tests/hello/e2e_test.py

import os
import sys
import subprocess

import pytest


@pytest.mark.e2e
def test_main_end_to_end_real_connection():
    """
    E2E real: executa `python -m src.hello.main` com LLM real.
    """

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("No API key found in environment, skipping real E2E test")

    result = subprocess.run(
        [sys.executable, "-m", "src.hello.main"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=True,
    )

    stdout = result.stdout
    lines = [line for line in stdout.splitlines() if line.strip()]

    assert len(lines) >= 2
    first_line = lines[0]
    last_line = lines[-1]

    assert last_line == "END"

    # Exemplo de saída atual:
    # greeting='Oi' news='A mosca ganhou a medalha olímpica.' chat='...'
    # Não é parseável facilmente, então checamos por "formato"
    assert "greeting=" in first_line
    assert "news=" in first_line
    assert "chat=" in first_line

    # Como o input é "Oi", é razoável esperar português/local language:
    assert "Oi" in first_line or "Olá" in first_line
