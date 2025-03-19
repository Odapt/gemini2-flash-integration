import os
import uuid
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image

from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp-image-generation", 
                 max_history: int = 30, output_dir: str = "output"):
        """
        Initialize the Gemini client with support for multi-turn chatting and image generation.
        
        Args:
            api_key: Your Google Gemini API key
            model: The model name to use
            max_history: Maximum number of messages to keep in history (sliding window)
            output_dir: Directory to save generated images
        """
        self.api_key = api_key
        self.model = model
        self.max_history = max_history
        self.output_dir = output_dir
        self.client = genai.Client(api_key=api_key)
        self.conversations = {}  # Store all active conversations
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Create a new conversation or reset an existing one.
        
        Args:
            conversation_id: Optional ID for the conversation. If None, a new ID is generated.
            
        Returns:
            The conversation ID
        """
        # Generate ID if not provided
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            
        # Create chat config
        config = types.GenerateContentConfig(
            response_modalities=['Text', 'Image']
        )
        
        # Create new chat session
        chat = self.client.chats.create(
            model=self.model,
            config=config
        )
        
        # Store the conversation
        self.conversations[conversation_id] = {
            "chat": chat,
            "history": [],
            "created_at": datetime.now(),
            "last_active": datetime.now()
        }
        
        return conversation_id
    
    def get_conversation_ids(self) -> List[str]:
        """Get all active conversation IDs."""
        return list(self.conversations.keys())
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the message history for a conversation."""
        if conversation_id not in self.conversations:
            return []
        return self.conversations[conversation_id]["history"]
    
    async def send_message(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """
        Send a message in a conversation and process the response.
        
        Args:
            conversation_id: The conversation ID
            message: The message text to send
            
        Returns:
            A dictionary with the response information
        """
        # Check if conversation exists, create if not
        if conversation_id not in self.conversations:
            conversation_id = self.create_conversation(conversation_id)
        
        conversation = self.conversations[conversation_id]
        chat = conversation["chat"]
        
        # Record user message in history
        user_message = {
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        conversation["history"].append(user_message)
        
        # Process sliding window if needed
        if len(conversation["history"]) > self.max_history:
            # Keep the most recent messages
            conversation["history"] = conversation["history"][-self.max_history:]
        
        # Update last active timestamp
        conversation["last_active"] = datetime.now()
        
        try:
            # Send message to Gemini
            response = chat.send_message(message)
            
            # Process response
            text_content = response.text if hasattr(response, 'text') else None
            image_paths = []
            
            # Extract and save any images
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content is not None:
                    if hasattr(candidate.content, 'parts'):
                        for i, part in enumerate(candidate.content.parts):
                            if hasattr(part, 'inline_data') and part.inline_data:
                                # Generate unique filename
                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                filename = f"{conversation_id}_{timestamp}_{i}.png"
                                
                                # Ensure output directory exists (redundant but safe)
                                os.makedirs(self.output_dir, exist_ok=True)
                                
                                filepath = os.path.join(self.output_dir, filename)
                                
                                # Save the image
                                image_data = part.inline_data.data
                                image = Image.open(BytesIO(image_data))
                                image.save(filepath)
                                
                                # Store the path
                                image_paths.append(filepath)
            
            # Create assistant message for history
            assistant_message = {
                "role": "assistant",
                "content": text_content,
                "images": image_paths,
                "timestamp": datetime.now().isoformat()
            }
            conversation["history"].append(assistant_message)
            
            return {
                "conversation_id": conversation_id,
                "text": text_content,
                "image_paths": image_paths,
                "success": True
            }
            
        except Exception as e:
            # Log the error and add to history
            error_message = {
                "role": "system",
                "content": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            conversation["history"].append(error_message)
            
            return {
                "conversation_id": conversation_id,
                "error": str(e),
                "success": False
            }
    
    def reset_conversation(self, conversation_id: str) -> bool:
        """
        Reset a conversation, clearing its history but keeping the ID.
        
        Returns:
            True if successful, False if the conversation doesn't exist
        """
        if conversation_id not in self.conversations:
            return False
            
        # Create a new chat session but keep the same ID
        self.create_conversation(conversation_id)
        return True
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation completely.
        
        Returns:
            True if successful, False if the conversation doesn't exist
        """
        if conversation_id not in self.conversations:
            return False
            
        del self.conversations[conversation_id]
        return True 