#!/usr/bin/env python
"""
Test script to simulate the radio flow with extensive logging.

This script simulates the complete radio flow from data generation to backward pass,
with detailed logging to help diagnose issues with the backward pass.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('radio_flow.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import components
from data_labelling.gradio_display.app import DanielsonArcherApp
from data_labelling.archer.backwardPass.danielson_model import DanielsonModel
from data_labelling.archer.database.supabase import SupabaseDatabase
from data_labelling.archer.archer import Archer
from data_labelling.archer.helpers.prompt import Prompt


def test_radio_flow(adalflow_enabled=True, run_optimization=True, verbose=False):
    """
    Test the complete radio flow with detailed logging.
    
    Args:
        adalflow_enabled (bool): Whether to enable AdaLFlow
        run_optimization (bool): Whether to run the optimization step
        verbose (bool): Whether to print additional debug information
    """
    logger.info("=== STARTING RADIO FLOW TEST ===")
    logger.info(f"AdaLFlow enabled: {adalflow_enabled}")
    
    if verbose:
        # Enable debug logging for all relevant modules
        logging.getLogger('data_labelling').setLevel(logging.DEBUG)
    
    try:
        # Initialize components
        logger.info("Initializing Danielson model")
        danielson_model = DanielsonModel(adalflow_enabled=adalflow_enabled)
        
        logger.info("Initializing database connection")
        # Use in-memory database for testing
        db = SupabaseDatabase(
            api_url=os.getenv("SUPABASE_API_URL", "http://localhost:6900"),
            api_key=os.getenv("SUPABASE_API_KEY", "admin.apikey")
        )
        
        logger.info("Initializing prompts")
        # Define initial prompts
        initial_prompts = [
            Prompt("Generate a comprehensive performance analysis and growth path for Danielson component {component_id} based on these low-inference notes: {input}"),
            Prompt("Create a detailed Danielson framework evaluation for component {component_id}. Analyze the teacher's performance using evidence from these notes: {input}")
        ]
        
        # Log the initial prompts
        for i, prompt in enumerate(initial_prompts):
            logger.info(f"Initial prompt {i+1}: {prompt.content[:100]}...")
        
        logger.info("Initializing Archer")
        # Initialize Archer with verbose logging
        archer = Archer(
            generator_model_name="gemini-2.0-flash",
            evaluator_model_name="gemini-2.0-flash",
            optimizer_model_name="gemini-2.0-flash",
            knowledge_base=["./data_labelling/eval"],
            rubric="""Evaluate the generated Danielson component summary on criteria including evidence-based analysis, framework alignment, clarity, actionability, and professionalism.""",
            initial_prompts=initial_prompts,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            human_validation_enabled=True,
            num_simulations_per_prompt=3,
            adalflow_enabled=adalflow_enabled,
            database_config={
                "api_url": os.getenv("SUPABASE_API_URL", "http://localhost:6900"),
                "api_key": os.getenv("SUPABASE_API_KEY", "admin.apikey")
            }
        )
        
        logger.info("Initializing Gradio app")
        # Initialize the Gradio app
        app = DanielsonArcherApp(
            archer_instance=archer,
            danielson_model=danielson_model,
            supabase_db=db
        )
        
        # Step 1: Generate input data
        logger.info("Step 1: Generating input data")
        low_inference_notes, component_id = app.generate_input_data()
        logger.info(f"Generated input data - Component: {component_id}")
        logger.debug(f"Low inference notes: {low_inference_notes[:100]}...")
        
        # Step 2: Generate summary
        logger.info("Step 2: Generating summary")
        result = app.generate_summary(low_inference_notes, component_id)
        logger.info(f"Generated summary - Score: {result['score']}")
        logger.debug(f"Content: {result['content'][:100]}...")
        logger.debug(f"Feedback: {result['feedback'][:100]}...")
        
        # Step 3: Save to database
        logger.info("Step 3: Saving to database")
        success = app.save_to_database(
            low_inference_notes=low_inference_notes,
            component_id=component_id,
            content=result["content"],
            score=result["score"],
            feedback=result["feedback"],
            perfect_output=result["perfect_output"]
        )
        logger.info(f"Save to database {'successful' if success else 'failed'}")
        
        # Log the current prompts before optimization
        logger.info("Current prompts before optimization:")
        prompt_text = app._get_current_prompts_text()
        for line in prompt_text.split('\n'):
            logger.info(f"  {line}")
        
        # Step 4: Run optimization (backward pass)
        if run_optimization:
            logger.info("Step 4: Running optimization (backward pass)")
            success = app.trigger_backward_pass()
            logger.info(f"Optimization {'successful' if success else 'failed'}")
            
            # Log the prompts after optimization
            logger.info("Prompts after optimization:")
            prompt_text = app._get_current_prompts_text()
            for line in prompt_text.split('\n'):
                logger.info(f"  {line}")
        else:
            logger.info("Skipping optimization step")
        
        logger.info("=== RADIO FLOW TEST COMPLETED ===")
        return True
        
    except Exception as e:
        logger.error(f"Error in radio flow test: {str(e)}", exc_info=True)
        return False


def main():
    """Parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test the radio flow with detailed logging.")
    parser.add_argument("--disable-adalflow", action="store_true", help="Disable AdaLFlow")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip the optimization step")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    test_radio_flow(
        adalflow_enabled=not args.disable_adalflow,
        run_optimization=not args.skip_optimization,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main() 