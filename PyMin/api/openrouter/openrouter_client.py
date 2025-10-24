import requests
import json
import os
from typing import Dict, List, Optional, Any


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-5-nano'):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.base_url = 'https://openrouter.ai/api/v1/chat/completions'
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/your-username/PyMin',
            'X-Title': 'PyMin'
        }
    

    def validate_messages_format(self, messages: List[Dict[str, str]]) -> tuple[bool, str]:
        if not isinstance(messages, list):
            return False, "Messages must be a list"
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                return False, f"Message at index {i} must be a dictionary"
            
            if 'role' not in message or 'content' not in message:
                return False, f"Message at index {i} must contain 'role' and 'content' keys"
            
            if not isinstance(message['role'], str) or not isinstance(message['content'], str):
                return False, f"Message at index {i} must have string values for 'role' and 'content'"
            
            if message['role'] not in ['system', 'user', 'assistant']:
                return False, f"Message at index {i} has invalid role '{message['role']}'. Must be 'system', 'user', or 'assistant'"
        
        return True, "Valid message format"
    

    def get_message_format_example(self) -> str:
        example = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "What's the weather like?"}
        ]
        explanation = (
            "Each message must be a dictionary with:\n"
            "- 'role': string ('system', 'user', or 'assistant')\n"
            "- 'content': string (the actual message content)"
        )
        import json
        return (
            "Correct message format example:\n"
            f"{json.dumps(example, indent=4)}\n\n"
            f"{explanation}"
        )


    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        is_valid, error_message = self.validate_messages_format(messages)
        if not is_valid:
            example = self.get_message_format_example()
            raise ValueError(f"Invalid message format: {error_message}\n\n{example}")
        
        data = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'stream': stream,
            **kwargs
        }
        
        if max_tokens is not None:
            data['max_tokens'] = max_tokens
        
        response = requests.post(self.base_url, headers=self.headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'OpenRouter API Error: {response.status_code}, {response.text}')
    

    def simple_chat(self, message: str) -> str:
        messages = [{'role': 'user', 'content': message}]
        response = self.chat_completion(messages)
        return response['choices'][0]['message']['content']
    

    def get_models(self) -> Dict[str, Any]:
        models_url = 'https://openrouter.ai/api/v1/models'
        response = requests.get(models_url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f'OpenRouter Models API Error: {response.status_code}, {response.text}')


def main():
    try:
        client = OpenRouterClient()
        
        print("Testing OpenRouter API...")
        response = client.simple_chat("What is the meaning of life?")
        print(f"Response: {response}")
        
        print("\nAvailable models:")
        models = client.get_models()
        for model in models.get('data', [])[:5]:
            print(f"- {model.get('id', 'Unknown')}: {model.get('name', 'Unknown')}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENROUTER_API_KEY environment variable")


if __name__ == "__main__":
    main()
