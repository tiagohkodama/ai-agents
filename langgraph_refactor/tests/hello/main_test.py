from langchain.agents.structured_output import ToolStrategy
from hello.schemas import GreetingsResponse
import hello.main as main_module


def test_main_creates_agent_and_invokes(monkeypatch, capsys):
    captured = {}

    def fake_init_chat_model(model_name, temperature, timeout, base_url, default_headers):
        # Capture args so we can assert on them later
        captured["init_chat_model"] = {
            "model_name": model_name,
            "temperature": temperature,
            "timeout": timeout,
            "base_url": base_url,
            "default_headers": default_headers,
        }
        return "fake-model"

    class DummyAgent:
        def __init__(self):
            self.invoked = False

        def invoke(self, input_):
            self.invoked = True
            captured["invoke_input"] = input_
            # Simulate a structured response compatible with GreetingsResponse
            return {
                "structured_response": {
                    "greeting": "Olá!",
                    "news": "The fly won the olimpic medal",
                    "chat": "Eu ouvi que a mosca ganhou uma medalha olímpica!",
                }
            }

    def fake_create_agent(model, system_prompt, tools, response_format):
        captured["create_agent"] = {
            "model": model,
            "system_prompt": system_prompt,
            "tools": tools,
            "response_format": response_format,
        }
        return DummyAgent()

    # Patch the functions in the hello.main module namespace
    monkeypatch.setattr(main_module, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(main_module, "create_agent", fake_create_agent)

    # Run
    main_module.main()

    # Assertions on init_chat_model
    init_args = captured["init_chat_model"]
    assert init_args["model_name"] == "gpt-4o-mini"
    assert init_args["temperature"] == 0.5
    assert init_args["timeout"] == 10
    # BASE_URL may be None, so we just check that the key exists
    assert "base_url" in init_args
    assert init_args["default_headers"] == {"FlowAgent": "Flow Api"}

    # Assertions on create_agent
    create_args = captured["create_agent"]
    assert create_args["model"] == "fake-model"
    # Should use the tools imported by main_module
    assert main_module.get_user_location in create_args["tools"]
    assert main_module.get_local_news in create_args["tools"]

    # Response format should be the ToolStrategy with your schema
    assert isinstance(create_args["response_format"], ToolStrategy)
    # No strict API here, but we can at least check the schema type
    assert GreetingsResponse.__name__ in repr(create_args["response_format"])

    # Assertions on invoke payload
    assert captured["invoke_input"] == {
        "messages": [{"role": "user", "content": "Oi"}]
    }

    # Check printed output
    out = capsys.readouterr().out.strip().splitlines()
    # First line: structured_response dict printed
    assert any("greeting" in line for line in out[0:1])
    # Last line should be the literal "END"
    assert out[-1] == "END"
