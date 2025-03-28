#!/usr/bin/env python
"""
test_end_to_end.py

This module provides an end-to-end test for the Archer system with Danielson components.
It simulates a complete user workflow from initialization through the Gradio interface
to content generation, evaluation, and prompt optimization.
"""

import os
import sys
import unittest
import time
import logging
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the parent directories are in the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import after path is set
from data_labelling.gradio_display.app import DanielsonArcherApp
from data_labelling.archer.backwardPass.danielson_model import DanielsonModel
from data_labelling.archer.database.supabase import SupabaseDatabase
from data_labelling.archer.archer import Archer
from data_labelling.archer.helpers.prompt import Prompt
from data_labelling.eval.danielson import normalize_score_integer


class TestEndToEnd(unittest.TestCase):
    """
    End-to-end test case for the Archer system with Danielson components.
    This test simulates a user interacting with the Gradio app from start to finish.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading environment variables.
        """
        # Determine the absolute path to the directory where this test file is located
        # The user indicated the .env file is in the same directory as this test
        test_dir = Path(__file__).parent.absolute()
        
        # Load environment variables from the .env file in the same directory as the test
        dotenv_path = test_dir / '.env'
        if not dotenv_path.exists():
            # Try looking in the root directory
            root_dir = Path(__file__).resolve().parent
            while root_dir.name and not (root_dir / '.env').exists():
                root_dir = root_dir.parent
            dotenv_path = root_dir / '.env'
        
        logger.info(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path)
        
        # Check for required environment variables
        required_vars = ['GOOGLE_API_KEY', 'SUPABASE_API_URL', 'SUPABASE_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Some tests may fail due to missing credentials.")

    def setUp(self):
        """
        Set up the test by initializing the Archer system and Gradio app.
        """
        # Initialize the database connection
        self.supabase_db = SupabaseDatabase(
            api_url=os.getenv("SUPABASE_API_URL"),
            api_key=os.getenv("SUPABASE_API_KEY")
        )
        
        # Connect to the database and initialize datasets
        connected = self.supabase_db.connect()
        self.assertTrue(connected, "Failed to connect to Argilla database")
        
        initialized = self.supabase_db.initialize_datasets()
        self.assertTrue(initialized, "Failed to initialize Argilla datasets")
        
        # Initialize the Danielson model
        self.danielson_model = DanielsonModel(adalflow_enabled=True)
        
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
        
        # Initialize Archer
        self.archer = Archer(
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
                "api_url": os.getenv("SUPABASE_API_URL"),
                "api_key": os.getenv("SUPABASE_API_KEY")
            }
        )
        
        # Initialize the Gradio app
        self.app = DanielsonArcherApp(
            archer_instance=self.archer,
            danielson_model=self.danielson_model,
            supabase_db=self.supabase_db
        )
        
        # Ensure we have a unique test identifier for this run
        self.test_run_id = f"test_run_{uuid.uuid4().hex[:8]}"
        
    def test_complete_user_flow(self):
        """
        Test the complete user flow from generating input to optimizing prompts.
        
        This test simulates:
        1. Generating sample input data
        2. Generating a summary for a component
        3. Saving the evaluation to the database
        4. Running the prompt optimization process
        5. Verifying the results
        """
        # Step 1: Generate sample input data
        logger.info("Step 1: Generating sample input data")
        low_inference_notes, component_id = self.app.generate_input_data()
        
        self.assertIsNotNone(low_inference_notes, "Failed to generate low inference notes")
        self.assertIsNotNone(component_id, "Failed to generate component ID")
        logger.info(f"Generated input for component {component_id}")
        
        # Step 2: Generate summary using the Archer system
        logger.info("Step 2: Generating summary")
        result = self.app.generate_summary(low_inference_notes, component_id)
        
        self.assertIsNotNone(result, "Failed to generate summary")
        self.assertIn("content", result, "Summary result missing 'content' field")
        self.assertIn("score", result, "Summary result missing 'score' field")
        self.assertIn("feedback", result, "Summary result missing 'feedback' field")
        
        content = result["content"]
        score = result["score"]
        feedback = result["feedback"]
        perfect_output = result.get("perfect_output", "")
        
        logger.info(f"Generated summary with score: {score}")
        
        # Step 3: Save the evaluation to the database
        logger.info("Step 3: Saving evaluation to database")
        saved = self.app.save_to_database(
            low_inference_notes=low_inference_notes,
            component_id=component_id,
            content=content,
            score=score,
            feedback=feedback,
            perfect_output=perfect_output
        )
        
        self.assertTrue(saved, "Failed to save evaluation to database")
        logger.info("Successfully saved evaluation to database")
        
        # Step 4: Simulate human validation by directly using the stored output_id
        logger.info("Step 4: Simulating human validation")
        
        # Get the output ID from the app's state (since we just saved it)
        output_id = self.app.current_output_id
        self.assertIsNotNone(output_id, "Failed to retrieve output ID")
        logger.info(f"Using output ID for human validation: {output_id}")
        
        # Adjust the score slightly to simulate human validation
        human_score = min(5, score + 1)  # Increment score but cap at 5
        human_feedback = f"The AI evaluation was good, but I'd like to see more specific evidence. Test ID: {self.test_run_id}"
        
        # Store human feedback
        human_update = self.supabase_db.store_human_feedback(
            output_id=output_id,
            score=human_score,
            feedback=human_feedback,
            improved_output=perfect_output
        )
        
        self.assertTrue(human_update, "Failed to store human feedback")
        logger.info("Successfully stored human feedback")
        
        # Step 5: Run the backward pass (prompt optimization)
        logger.info("Step 5: Running backward pass")
        optimization_result = self.app.trigger_backward_pass()
        
        self.assertTrue(optimization_result, "Failed to run backward pass")
        logger.info("Successfully ran backward pass")
        
        # Step 6: Verify the results
        logger.info("Step 6: Verifying results")
        
        # Get the current best prompts
        current_prompts = self.app._get_current_prompts_text()
        self.assertIsNotNone(current_prompts, "Failed to retrieve current prompts")
        logger.info("Current prompts retrieved successfully")
        
        # Verify that there are active prompts in the system
        active_prompts = self.archer.active_generator_prompts
        self.assertIsNotNone(active_prompts, "No active prompts found")
        self.assertGreater(len(active_prompts), 0, "No active prompts found")
        logger.info(f"Found {len(active_prompts)} active prompts")
        
        # Check if performance metrics are available
        metrics = self.supabase_db.get_performance_metrics()
        self.assertIsNotNone(metrics, "Failed to retrieve performance metrics")
        logger.info("Performance metrics retrieved successfully")
        
        # Run a new generation with the optimized prompts to ensure they work
        logger.info("Step 7: Testing optimized prompts with new input")
        new_notes, new_component = self.app.generate_input_data()
        new_result = self.app.generate_summary(new_notes, new_component)
        
        self.assertIsNotNone(new_result, "Failed to generate summary with optimized prompts")
        self.assertIn("content", new_result, "New summary result missing 'content' field")
        self.assertIn("score", new_result, "New summary result missing 'score' field")
        
        new_content = new_result["content"]
        new_score = new_result["score"]
        
        logger.info(f"Generated new summary with score: {new_score}")
        logger.info("End-to-end test completed successfully")

    def tearDown(self):
        """
        Clean up any resources after the test.
        """
        # No specific cleanup needed as the database connection will be closed automatically
        pass


if __name__ == "__main__":
    unittest.main() 