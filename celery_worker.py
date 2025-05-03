from celery import Celery
import os
import logging
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.graph_stores.nebula import NebulaPropertyGraphStore
from pdf2image import convert_from_path
import requests
import json
import base64
import nest_asyncio
nest_asyncio.apply()
import time
import uuid
from config.config import props_schema, VECTOR_DB_DIMENSION, OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBED_MODEL, OPENAI_VISION_MODEL
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a stream handler (console output)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class CustomTextNode(TextNode):
    """
    A custom subclass of TextNode that ensures a unique document ID is present.
    This fixes the error:
      "BaseModel.__init__() takes 1 positional argument but 3 were given"
    by only passing keyword arguments to the parent __init__.
    """
    def __init__(self, **data: Any) -> None:
        # Ensure metadata exists and is a dict
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = dict(metadata)
        # Generate and add a doc_id if not already present
        if "doc_id" not in metadata:
            metadata["doc_id"] = str(uuid.uuid4())
        data["metadata"] = metadata
        
        # Now call the parent initializer using keyword arguments only
        super().__init__(**data)

# Initialize the LLM with OpenAI
Settings.llm = OpenAI(
    model=OPENAI_CHAT_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.3
)

# Define embedding model with OpenAI
Settings.embed_model = OpenAIEmbedding(
    model=OPENAI_EMBED_MODEL,
    api_key=OPENAI_API_KEY,
)

# Configure Celery with Redis as broker
celery_app = Celery("worker", broker="redis://localhost:6379/0", backend="redis://localhost:6379/1")

# Initialize a node parser for document processing
node_parser = SimpleNodeParser.from_defaults()

def get_base64_encoded_image(image_path):
    """Convert an image to base64 encoding for API requests"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_with_openai(image_path, prompt):
    """Process an image using OpenAI's vision model"""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        with open(image_path, "rb") as image_file:
            response = client.chat.completions.create(
                model=OPENAI_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{get_base64_encoded_image(image_path)}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error processing image with OpenAI: {str(e)}")

@celery_app.task
def process_file_for_training(file_location: str, user_id: int, repository_id: int):
    """
    Process uploaded files for training and indexing based on file type.
    This function handles different file types and creates appropriate indexes.
    """
    logger.info("Starting processing for file: %s", file_location)
    try:
        # Extract filename and extension
        filename = os.path.basename(file_location)
        temp_dir_name, file_extension = os.path.splitext(filename)
        repo_upload_dir = os.path.dirname(file_location)
             
        property_graph_store = NebulaPropertyGraphStore(
            space=f'space_{user_id}',
            props_schema=props_schema,
        )

        # Initialize index and vector stores
        index_config = {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 128
            }
        }

        graph_vec_store = MilvusVectorStore(
            uri="http://localhost:19530", 
            collection_name=f"space_{user_id}",
            dim=VECTOR_DB_DIMENSION, 
            overwrite=False,
            similarity_metric="COSINE",
            index_config=index_config,
        )
        
        graph_index = PropertyGraphIndex.from_existing(
            property_graph_store=property_graph_store,
            vector_store=graph_vec_store,
            llm=Settings.llm,
            embed_model=Settings.embed_model,
        )
        
        # Vision model prompt
        text_con_prompt = """
            Please analyze the provided image and generate a detailed, plain-language description of its contents. 
            Include key elements such as objects, people, colors, spatial relationships, background details, and any text visible in the image. 
            The goal is to create a comprehensive textual representation that fully conveys the visual information to someone who cannot see the image.
        """

        source_data = SimpleDirectoryReader(input_files=[file_location]).load_data()
        logger.info(f"Source data metadata: {source_data[0].metadata}")
        
        # Create a thread-local event loop policy
        try:
            # Process based on file type
            match file_extension: 
                case '.txt':
                    logger.info("Starting text file processing............")
                    simple_doc = SimpleDirectoryReader(input_files=[file_location]).load_data()
                    logger.info("Simple doc has been loaded")
                    
                    for doc in simple_doc:
                        # Use the standard insert method but catch and handle event loop errors
                        try:
                            # PropertyGraphIndex doesn't have node_parser, use the regular insert method
                            # but inside try/except to handle any event loop issues
                            graph_index.insert(doc)
                        except RuntimeError as e:
                            if "Event loop is closed" in str(e):
                                # Recreate the index with a new event loop
                                import asyncio
                                new_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(new_loop)
                                graph_index.insert(doc)
                            else:
                                raise
                
                case '.pdf':
                    logger.info("Starting PDF file processing")
                    
                    # Step 1: Extract text from PDF using the SimpleDirectoryReader
                    pdf_docs = SimpleDirectoryReader(input_files=[file_location]).load_data()
                    logger.info(f"Loaded {len(pdf_docs)} documents from PDF")
                    
                    for doc in pdf_docs:
                        try:
                            graph_index.insert(doc)
                            logger.info(f"Inserted text from PDF document: {doc.metadata}")
                        except Exception as e:
                            logger.error(f"Error inserting PDF text: {str(e)}")
                    
                    # Step 2: Convert PDF to images for visual processing
                    try:
                        temp_img_dir = os.path.join(repo_upload_dir, temp_dir_name)
                        os.makedirs(temp_img_dir, exist_ok=True)
                        
                        logger.info(f"Converting PDF to images in {temp_img_dir}")
                        images = convert_from_path(file_location)
                        
                        for i, image in enumerate(images):
                            img_path = os.path.join(temp_img_dir, f'page_{i+1}.jpg')
                            image.save(img_path, 'JPEG')
                            
                            # Process each image with vision model
                            logger.info(f"Processing image {img_path} with OpenAI vision model")
                            img_description = process_image_with_openai(img_path, text_con_prompt)
                            
                            # Create a document from the image description
                            img_doc = Document(
                                text=img_description,
                                metadata={
                                    "file_name": filename,
                                    "page_number": i+1,
                                    "file_type": "pdf_image",
                                    "source": file_location,
                                    "doc_id": f"{filename}_img_{i+1}_{uuid.uuid4()}"
                                }
                            )
                            
                            # Insert the image document
                            graph_index.insert(img_doc)
                            logger.info(f"Inserted PDF image content for page {i+1}")
                            
                    except Exception as e:
                        logger.error(f"Error processing PDF images: {str(e)}")
                
                case '.jpg' | '.jpeg' | '.png':
                    logger.info("Starting image file processing")
                    
                    try:
                        # Process the image with vision model
                        img_description = process_image_with_openai(file_location, text_con_prompt)
                        
                        # Create a document from the image description
                        img_doc = Document(
                            text=img_description,
                            metadata={
                                "file_name": filename,
                                "file_type": "image",
                                "source": file_location,
                                "doc_id": f"{filename}_{uuid.uuid4()}"
                            }
                        )
                        
                        # Insert the image document
                        graph_index.insert(img_doc)
                        logger.info(f"Inserted image content for {filename}")
                        
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                
                case _:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in process_file_for_training: {str(e)}")
        raise

    logger.info(f"Completed processing for file: {file_location}")
    return {"status": "success", "file": file_location}