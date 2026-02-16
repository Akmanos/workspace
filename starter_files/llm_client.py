from typing import Dict, List
from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    persona = f"""You are a NASA mission expert assistant.

Guidelines:
- Provide clear, concise answers.
- Use ONLY the provided CONTEXT to answer factual questions. If the CONTEXT does not contain the answer, say you don't know.
- When helpful, quote or paraphrase the relevant part of the CONTEXT.
- Always cite sources using the bracketed source numbers from the CONTEXT headers, e.g. [1], [2].
- Explicitly state when you encounter conflicting information in the CONTEXT.

CONTEXT:
{context}
"""
    # Set context in messages
    messages: List[Dict[str, str]] = [{"role": "system", "content": persona}]

    # Keep only user/assistant conversation history to avoid duplicating system prompts
    for m in conversation_history[-10:]:
        if m.get("role") in {"user", "assistant"} and "content" in m:
            messages.append({"role": m["role"], "content": m["content"]})

    # Add current message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI Client
    try:
        openai_client = OpenAI(
            base_url="https://openai.vocareum.com/v1", api_key=openai_key
        )

        # Send request to OpenAI
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"
