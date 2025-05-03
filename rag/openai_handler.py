import os
import json
import aiohttp
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from config.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL

# System prompt for authentication
openai_system_prompt = """
You are an authentication assistant. Your task is to extract exactly three pieces of information from the user:
1. Action: 'sign-up' or 'sign-in'
2. Phone number
3. Password

Respond in JSON format with three fields:
{
    "action": "sign-up" or "sign-in",
    "phone_number": "extracted phone number",
    "password": "extracted password",
    "instruction": "your response to the user"
}

If any information is missing or unclear, respond with appropriate instructions to guide the user. 
Always use the "instruction" field to communicate with the user.
"""

class OpenAIHandler:
    def __init__(self, system_prompt: str = "", api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def agenerate_chat_completion(self, messages, model=OPENAI_CHAT_MODEL):
        """
        Generate a chat completion using OpenAI's API
        
        Args:
            messages: List of message objects with role and content
            model: The OpenAI model to use
            
        Returns:
            The content of the assistant's response
        """
        formatted_messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            formatted_messages.append({"role": "system", "content": self.system_prompt})
        
        # Format messages for OpenAI API
        for message in messages:
            if isinstance(message, dict) and "role" in message and "content" in message:
                formatted_messages.append(message)
            else:
                # Assume user message if format is unclear
                formatted_messages.append({"role": "user", "content": str(message)})
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating chat completion: {str(e)}")
    
    async def aemb_text(self, text, model=OPENAI_EMBED_MODEL):
        """
        Generate embeddings for the given text using OpenAI's API
        
        Args:
            text: The text to embed
            model: The embedding model to use
            
        Returns:
            The embedding vector
        """
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}") 