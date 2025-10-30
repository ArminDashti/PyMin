import requests
import json
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from ...util.image_converter import validate_image_path, encode_image_to_base64


class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, model: str = 'gpt-4o-mini'):
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


    def is_vision_model(self, model: Optional[str] = None) -> bool:
        """Check if the specified model supports vision/image processing."""
        model_to_check = model or self.model
        return any(vision_model in model_to_check.lower() for vision_model in self.vision_models)


    def validate_image_path(self, image_path: Union[str, Path]) -> tuple[bool, str]:
        """Validate if the image path exists and is in a supported format."""
        return validate_image_path(image_path)


    def encode_image_to_base64(self, image_path: Union[str, Path]) -> tuple[bool, str, str]:
        """
        Encode an image file to base64 format.
        
        Returns:
            tuple: (success: bool, data: str, mime_type: str)
        """
        return encode_image_to_base64(image_path)


    def create_image_message(self, image_path: Union[str, Path], text: str = "") -> Dict[str, Any]:
        """
        Create a message with image content for multimodal chat completion.
        
        Args:
            image_path: Path to the image file
            text: Optional text to accompany the image
            
        Returns:
            Dictionary containing the message with image data
        """
        success, image_data, mime_type = self.encode_image_to_base64(image_path)
        if not success:
            raise ValueError(f"Failed to encode image: {image_data}")
        
        message = {
            "role": "user",
            "content": []
        }
        
        if text:
            message["content"].append({
                "type": "text",
                "text": text
            })
        
        message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}"
            }
        })
        
        return message


    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "Describe what you see in this image.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze an image using a vision-capable model.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for image analysis
            model: Model to use (defaults to instance model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API call
            
        Returns:
            API response dictionary
        """
        model_to_use = model or self.model
        
        # Check if model supports vision
        if not self.is_vision_model(model_to_use):
            raise ValueError(f"Model '{model_to_use}' does not support image processing. "
                           f"Use one of these vision models: {', '.join(self.vision_models)}")
        
        # Create image message
        image_message = self.create_image_message(image_path, prompt)
        
        # Prepare messages for API call
        messages = [image_message]
        
        # Use the existing chat_completion method with the image message
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_to_use,
            **kwargs
        )


    def analyze_multiple_images(
        self,
        image_paths: List[Union[str, Path]],
        prompt: str = "Analyze these images and describe what you see.",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze multiple images using a vision-capable model.
        
        Args:
            image_paths: List of paths to image files
            prompt: Text prompt for image analysis
            model: Model to use (defaults to instance model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters for the API call
            
        Returns:
            API response dictionary
        """
        if not image_paths:
            raise ValueError("At least one image path must be provided")
        
        model_to_use = model or self.model
        
        # Check if model supports vision
        if not self.is_vision_model(model_to_use):
            raise ValueError(f"Model '{model_to_use}' does not support image processing. "
                           f"Please use a vision-capable model like gpt-4o, claude-3, or gemini-pro-vision.")
        
        # Create message with multiple images
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
        
        # Add each image to the message
        for image_path in image_paths:
            success, image_data, mime_type = self.encode_image_to_base64(image_path)
            if not success:
                raise ValueError(f"Failed to encode image {image_path}: {image_data}")
            
            message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }
            })
        
        # Use the existing chat_completion method
        return self.chat_completion(
            messages=[message],
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_to_use,
            **kwargs
        )


    def simple_image_analysis(self, image_path: Union[str, Path], prompt: str = "Describe what you see in this image.") -> str:
        """
        Simple image analysis that returns just the text response.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for image analysis
            
        Returns:
            String response from the model
        """
        response = self.analyze_image(image_path, prompt)
        return response['choices'][0]['message']['content']


    def get_vision_models(self) -> List[str]:
        """Get a list of available vision-capable models."""
        try:
            models_response = self.get_models()
            vision_models = []
            
            for model in models_response.get('data', []):
                model_id = model.get('id', '').lower()
                if any(vision_model in model_id for vision_model in self.vision_models):
                    vision_models.append(model.get('id', ''))
            
            return vision_models
        except Exception as e:
            print(f"Error fetching vision models: {e}")
            return list(self.vision_models)


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
        
        print("\nVision-capable models:")
        vision_models = client.get_vision_models()
        for model in vision_models[:5]:
            print(f"- {model}")
        
        # Example of image processing (uncomment to test with actual image)
        # print("\nTesting image analysis...")
        # try:
        #     # Replace with actual image path
        #     image_path = "path/to/your/image.jpg"
        #     if os.path.exists(image_path):
        #         analysis = client.simple_image_analysis(image_path, "What do you see in this image?")
        #         print(f"Image analysis: {analysis}")
        #     else:
        #         print("No test image found. Place an image file to test image analysis.")
        # except Exception as img_error:
        #     print(f"Image analysis error: {img_error}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set your OPENROUTER_API_KEY environment variable")


if __name__ == "__main__":
    main()
