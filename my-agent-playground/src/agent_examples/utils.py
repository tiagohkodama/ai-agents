from typing import Any, Dict

def pretty_print_agent_response(response: Dict[str, Any]):
    """
    Pretty print structured LangChain agent output in a human-friendly format.
    Handles messages, tool calls, and usage metadata.
    """

    messages = response.get("messages", [])
    print("\n================= AGENT RESULT =================")

    # 1. Final assistant answer (last AIMessage)
    final_answer = None
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            final_answer = msg.content
            break
    
    if final_answer:
        print("\n🤖 Assistant Final Answer:")
        print(final_answer)
    else:
        print("\n🤖 Assistant returned no final text.")

    # 2. Tool calls
    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    if tool_calls:
        print("\n🔧 Tools Used:")
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args", {})
            print(f"   - {name}({args})")

    # 3. Token usage
    ai_msg_meta = None
    for msg in reversed(messages):
        if hasattr(msg, "response_metadata") and msg.response_metadata:
            ai_msg_meta = msg.response_metadata
            break

    if ai_msg_meta and "token_usage" in ai_msg_meta:
        usage = ai_msg_meta["token_usage"]
        print("\n📊 Token Usage:")
        print(f"   - prompt:     {usage.get('prompt_tokens', '?')}")
        print(f"   - completion: {usage.get('completion_tokens', '?')}")
        print(f"   - total:      {usage.get('total_tokens', '?')}")

    print("\n===============================================\n")
