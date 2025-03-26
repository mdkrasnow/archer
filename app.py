#!/usr/bin/env python
"""
app.py

This script sets up and launches the Archer-Danielson framework application.
It handles the configuration and instantiation of all necessary components.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the parent directory is in the path for imports
sys.path.append(str(Path(__file__).parent))

# Import the application
from gradio_display.app import DanielsonArcherApp
from archer.backwardPass.danielson_model import DanielsonModel
from archer.database.argilla import ArgillaDatabase
from archer.archer import Archer
from archer.helpers.prompt import Prompt

def load_environment():
    """
    Load environment variables from .env file.
    """
    # Try to load from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)

    # Check for required environment variables
    required_vars = ['OPENROUTER_API_KEY', 'ARGILLA_API_URL', 'ARGILLA_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some functionality may be limited.")

def initialize_archer():
    """
    Initialize the Archer instance with the Danielson model.
    
    Returns:
        Archer: The initialized Archer instance
    """
    # Initialize the Danielson model
    danielson_model = DanielsonModel(adalflow_enabled=True)
    
    # Define initial prompts for Archer
    initial_prompts = [
        Prompt("Generate a comprehensive performance analysis and growth path for Danielson component {component_id} based on these low-inference notes: {input}"),
        Prompt("Create a detailed Danielson framework evaluation for component {component_id}. Analyze the teacher's performance using evidence from these notes: {input}")
    ]
    
    # Define the rubric
    rubric = """
    Evaluate the generated Danielson component summary on the following criteria:
    
    1. Evidence-Based Analysis (1-5): Does the summary include specific evidence from the low-inference notes? 
       Are direct quotes used to support observations?
       
    2. Framework Alignment (1-5): Does the summary correctly interpret the evidence according to the Danielson 
       framework's expectations for this specific component?
       
    3. Clarity and Structure (1-5): Is the summary well-organized with clear separation between performance 
       analysis and growth path sections?
       
    4. Actionability (1-5): Are the growth recommendations specific, concrete, and implementable? Do they focus 
       on high-leverage changes that would improve student learning outcomes?
       
    5. Professionalism (1-5): Is the language balanced, constructive, and appropriate for a professional 
       evaluation context?
       
    Give an overall score from 1-5, with 5 being the highest quality. Provide specific feedback on strengths 
    and areas for improvement, and include an example of what a perfect summary would look like for this input.
    """
    
    # Initialize Archer with the Danielson model
    archer = Archer(
        generator_model_name="gemini-2.0-flash",
        evaluator_model_name="gemini-2.0-flash",
        optimizer_model_name="gemini-2.0-flash",
        knowledge_base=["./data_labelling/eval"],  # Path to knowledge directories
        rubric=rubric,
        initial_prompts=initial_prompts,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        human_validation_enabled=True,
        num_simulations_per_prompt=3,
        database_config={
            "api_url": os.getenv("ARGILLA_API_URL", "http://localhost:6900"),
            "api_key": os.getenv("ARGILLA_API_KEY", "admin.apikey")
        }
    )
    
    return archer

def main():
    """
    Main function to parse arguments and launch the application.
    """
    parser = argparse.ArgumentParser(description="Run the Archer-Danielson framework application")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--no-archer", action="store_true", help="Run without Archer integration (lighter mode)")
    args = parser.parse_args()
    
    # Load environment variables
    load_environment()
    
    # Initialize components
    archer_instance = None if args.no_archer else initialize_archer()
    argilla_db = ArgillaDatabase(
        api_url=os.getenv("ARGILLA_API_URL", "http://localhost:6900"),
        api_key=os.getenv("ARGILLA_API_KEY", "admin.apikey")
    )
    
    # Initialize the application
    app = DanielsonArcherApp(
        archer_instance=archer_instance,
        argilla_db=argilla_db
    )
    
    # Launch the application
    logger.info(f"Launching application on port {args.port}")
    app.launch(share=args.share, server_port=args.port)

if __name__ == "__main__":
    main() 