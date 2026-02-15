from typing import Dict, List
from openai import OpenAI


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    persona = f"""You are a NASA expert.

Guidelines:
- Provide clear, concise answers
- Highlight when you do not have sufficient information to accurately answer a question
- Explicitly state when you encounter conflicting information.

CONTEXT:
{context}
"""
    # Set context in messages
    conversation_history.append(
        {
            "role": "system",
            "content": persona
        }
    )
    # Add chat history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    # Create OpenAI Client
    try:
        openai_client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=openai_key
        )
        # Send request to OpenAI
        response = openai_client.chat.completions.create(
            model=model,
            messages=conversation_history,
            temperature=0,
            max_completion_tokens=200,
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(str(e))
        return None
