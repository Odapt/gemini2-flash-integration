import os
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
from dotenv import load_dotenv

load_dotenv('.env')

from gemini_client import GeminiClient

# Initialize FastAPI app
app = FastAPI(title="Gemini Chat API", 
              description="API for interacting with Google's Gemini model with support for image generation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - with better error handling for API key
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY environment variable is not set")
    print("Please set it with: export GEMINI_API_KEY='your-api-key'")
    sys.exit(1)

OUTPUT_DIR = os.environ.get("GEMINI_OUTPUT_DIR", "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the Gemini client
gemini_client = GeminiClient(api_key=API_KEY, output_dir=OUTPUT_DIR)

# Serve the output directory for image viewing
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# Pydantic models for request/response validation
class MessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ConversationResponse(BaseModel):
    conversation_id: str
    history: List[Dict[str, Any]]

class MessageResponse(BaseModel):
    conversation_id: str
    text: Optional[str] = None
    image_urls: Optional[List[str]] = None
    success: bool
    error: Optional[str] = None

# Helper function to convert file paths to URLs
def convert_paths_to_urls(base_url: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert image file paths in history to URLs."""
    updated_history = []
    
    for message in history:
        message_copy = message.copy()
        
        # Convert image paths to URLs if they exist
        if "images" in message_copy and message_copy["images"]:
            image_urls = []
            for img_path in message_copy["images"]:
                if img_path:
                    # Extract just the filename from the path
                    filename = os.path.basename(img_path)
                    image_urls.append(f"{base_url}/images/{filename}")
            
            message_copy["image_urls"] = image_urls
            del message_copy["images"]  # Remove the original paths
            
        updated_history.append(message_copy)
        
    return updated_history

# API Endpoints
@app.post("/chat", response_model=MessageResponse)
async def chat(request: MessageRequest):
    """Send a message to Gemini and get a response."""
    response = await gemini_client.send_message(
        conversation_id=request.conversation_id,
        message=request.message
    )
    
    # Convert file paths to URLs for the response
    image_urls = []
    for path in response.get("image_paths", []):
        if path:
            filename = os.path.basename(path)
            image_urls.append(f"/images/{filename}")
    
    return {
        "conversation_id": response["conversation_id"],
        "text": response.get("text"),
        "image_urls": image_urls,
        "success": response.get("success", False),
        "error": response.get("error")
    }

@app.get("/conversations", response_model=List[str])
def list_conversations():
    """Get all active conversation IDs."""
    return gemini_client.get_conversation_ids()

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(conversation_id: str, base_url: str = Query("http://localhost:8000")):
    """Get the full history of a conversation with image URLs."""
    history = gemini_client.get_conversation_history(conversation_id)
    
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Convert image paths to URLs
    updated_history = convert_paths_to_urls(base_url, history)
    
    return {
        "conversation_id": conversation_id,
        "history": updated_history
    }

@app.post("/conversations/{conversation_id}/reset")
def reset_conversation(conversation_id: str):
    """Reset a conversation, clearing its history."""
    success = gemini_client.reset_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"status": "success", "message": "Conversation reset"}

@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation completely."""
    success = gemini_client.delete_conversation(conversation_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"status": "success", "message": "Conversation deleted"}

@app.get("/images/{filename}")
def get_image(filename: str):
    """Get a specific image by filename."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(file_path)

# Main entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 