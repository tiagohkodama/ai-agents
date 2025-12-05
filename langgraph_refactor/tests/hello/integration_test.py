import ast
from hello.schemas import GreetingsResponse
import hello.main as app


def test_main_integration(monkeypatch, capsys):

    def fake_init_chat_model(*args, **kwargs):
        return object()

    class DummyAgent:
        def __init__(self):
            self.invoked = False
            self.last_input = None

        def invoke(self, data):
            self.invoked = True
            self.last_input = data
            return {
                "structured_response": {
                    "greeting": "Mano!",
                    "news": "The horse won the elections",
                    "chat": "Eu ouvi que o cavalo ganhou as eleições!",
                }
            }

    dummy_agent = DummyAgent()

    def fake_create_agent(model, system_prompt, tools, response_format):
        return dummy_agent

    monkeypatch.setattr(app, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(app, "create_agent", fake_create_agent)

    app.main()

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]

    assert len(lines) >= 2
    first_line = lines[0]
    last_line = lines[-1]

    assert last_line == "END"

    expected_dict = {
        "greeting": "Mano!",
        "news": "The horse won the elections",
        "chat": "Eu ouvi que o cavalo ganhou as eleições!",
    }

    data = ast.literal_eval(first_line)

    assert data == expected_dict

    resp = GreetingsResponse(**data)
    expected_model = GreetingsResponse(**expected_dict)
    assert resp == expected_model

    assert dummy_agent.invoked is True
    assert dummy_agent.last_input == {
        "messages": [{"role": "user", "content": "Oi"}]
    }
