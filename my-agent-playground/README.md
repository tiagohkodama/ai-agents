# AI Agents Playground

This repository is dedicated to studying **AI agents**, primarily using:

* **Python**
* **LangChain**
* **LangGraph**
* **OpenAI models**

The goal is to explore different types of agents, tools, multi-agent workflows, memory, and reasoning patterns.

---

## Project Structure

```
my-agent-playground/
├─ src/
│  └─ agent_examples/
│     ├─ example_agent.py     # minimal agent that calls Python tools
│     ├─ tools.py             # functions exposed as agent tools
│     ├─ utils.py             # helpers for printing/debugging
│     └─ __init__.py
├─ requirements.txt
├─ .env
└─ README.md
```

---

## Running the agent

Make sure your virtual environment is activated and you have a valid `OPENAI_API_KEY` in your `.env`.

Then execute:

```bash
python -m src.agent_examples.example_agent
```

---

## Notes

* Use the repository to add more examples and experiments.
* Each agent variation should be placed in `src/agent_examples/` as a separate module.
* Keep tools modular so they can be reused across agents.
