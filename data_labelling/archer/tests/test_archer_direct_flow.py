#!/usr/bin/env python
"""
test_archer_direct_flow.py

This module provides a direct end-to-end test for the Archer system components
without using the Gradio interface. It tests the core functionality of the system
including forward pass, backward pass, and database operations directly.
"""

import os
import sys
import unittest
import json
import uuid
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the parent directories are in the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import after path is set
from data_labelling.archer.archer import Archer
from data_labelling.archer.database.supabase import SupabaseDatabase
from data_labelling.archer.helpers.prompt import Prompt
from data_labelling.archer.backwardPass.danielson_model import DanielsonModel


# Sample low-inference notes for testing
SAMPLE_LOW_INFERENCE_NOTES = """
The teacher begins class by welcoming students and displaying the day's agenda on the smart board. 
Students enter and take their seats quietly. After taking attendance, the teacher reviews yesterday's homework 
assignment on fractions. "Who can tell me how to add fractions with unlike denominators?" she asks. 
Several students raise their hands, and she calls on a student in the front row. The student explains the process, 
and the teacher affirms the answer while writing the steps on the whiteboard.

The classroom is arranged with desks in groups of four. There are anchor charts visible on the walls displaying 
key vocabulary terms and mathematical concepts. The teacher moves between groups, checking progress and asking 
probing questions. "What strategy are you using to solve this problem?" she asks one group. "How did you know to 
apply that approach?" she asks another. Students are engaged in collaborative work, discussing their thinking and 
writing in their math journals.
"""

class TestArcherDirectFlow(unittest.TestCase):
    """
    Direct end-to-end test case for the Archer system components.
    Tests the core functionality without using the Gradio interface.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by loading environment variables.
        """
        # Load environment variables
        dotenv_path = Path(__file__).parent.parent.parent.parent / '.env'
        load_dotenv(dotenv_path=dotenv_path)
        
        # Check for required environment variables
        required_vars = ['OPENROUTER_API_KEY', 'SUPABASE_API_URL', 'SUPABASE_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.warning("Some tests may fail due to missing credentials.")

    def setUp(self):
        """
        Set up the test by initializing the Archer system components.
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
        
        # Initialize Archer with human validation disabled for direct testing
        self.archer = Archer(
            generator_model_name="gemini-2.0-flash",
            evaluator_model_name="gemini-2.0-flash",
            optimizer_model_name="gemini-2.0-flash",
            knowledge_base=["./data_labelling/eval"],  # Path to knowledge directories
            rubric=rubric,
            initial_prompts=initial_prompts,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            human_validation_enabled=False,  # Disable for direct testing
            num_simulations_per_prompt=2,  # Reduce for faster testing
            database_config={
                "api_url": os.getenv("SUPABASE_API_URL"),
                "api_key": os.getenv("SUPABASE_API_KEY")
            }
        )
        
        # Ensure we have a unique test identifier for this run
        self.test_run_id = f"direct_test_{uuid.uuid4().hex[:8]}"
        
    def test_direct_archer_flow(self):
        """
        Test the direct flow of the Archer system components.
        
        This test:
        1. Prepares input data
        2. Runs the forward pass
        3. Simulates human evaluation by updating database records
        4. Runs the backward pass
        5. Verifies that prompt optimization occurred
        """
        # Step 1: Prepare input data for component 2a (Classroom Culture)
        component_id = "2a"
        input_data = {
            "input": SAMPLE_LOW_INFERENCE_NOTES,
            "component_id": component_id
        }
        
        logger.info(f"Step 1: Prepared input data for component {component_id}")
        
        # Step 2: Run the forward pass to generate content and evaluations
        logger.info("Step 2: Running forward pass")
        evaluations = self.archer.run_forward_pass(input_data)
        
        # Verify we got results
        self.assertIsNotNone(evaluations, "Forward pass returned None")
        self.assertGreater(len(evaluations), 0, "No evaluations generated")
        
        # Extract data from evaluations
        evaluation_records = []
        for prompt, content, eval_result in evaluations:
            self.assertIsNotNone(content, "Generated content is None")
            self.assertGreater(len(content), 0, "Generated content is empty")
            self.assertIsNotNone(eval_result, "Evaluation result is None")
            self.assertIn("score", eval_result, "Evaluation missing score")
            
            logger.info(f"Generated content with score: {eval_result.get('score')}")
            
            # Store the evaluation record for later use
            evaluation_records.append({
                "prompt": prompt,
                "content": content,
                "evaluation": eval_result
            })
        
        # Step 3: Manually store content and evaluations in the database
        logger.info("Step 3: Storing content and evaluations in database")
        stored_output_ids = []
        
        for record in evaluation_records:
            content = record["content"]
            eval_result = record["evaluation"]
            prompt = record["prompt"]
            
            # Store the output
            output_id = self.supabase_db.store_generated_content(
                input_data=json.dumps(input_data),
                content=content,
                prompt_id=getattr(prompt, "id", str(uuid.uuid4())),
                round_num=1  # Assuming first round
            )
            
            self.assertIsNotNone(output_id, "Failed to store generated content")
            
            # Store the evaluation
            eval_stored = self.supabase_db.store_evaluation(
                output_id=output_id,
                score=eval_result.get("score", 3),
                feedback=eval_result.get("feedback", "No feedback provided"),
                improved_output=eval_result.get("improved_output", ""),
                is_human=False  # This is an AI evaluation
            )
            
            self.assertTrue(eval_stored, "Failed to store evaluation")
            
            # Store the output ID for later use
            stored_output_ids.append(output_id)
        
        logger.info(f"Stored {len(stored_output_ids)} outputs with evaluations")
        
        # Step 4: Simulate human validation by updating some evaluations
        logger.info("Step 4: Simulating human validation")
        
        # Update at least one evaluation with human feedback
        if stored_output_ids:
            output_id = stored_output_ids[0]
            human_feedback = f"This evaluation needs more specific examples. Test ID: {self.test_run_id}"
            human_score = 4  # Good but not perfect
            
            # Create an improved output
            improved_output = "**Performance Analysis**\n\nThe teacher demonstrates proficient classroom culture management as evidenced by the well-organized physical space with \"desks in groups of four\" and the smooth transition at the beginning of class where \"students enter and take their seats quietly.\" The teacher effectively uses questioning strategies to engage students, asking \"What strategy are you using to solve this problem?\" and \"How did you know to apply that approach?\" These questions promote critical thinking and demonstrate respect for student intellect.\n\n**Growth Path**\n\nTo strengthen classroom culture further, the teacher should consider implementing more explicit student-led discussions where peers evaluate each other's mathematical reasoning. This could be accomplished by establishing a structured protocol for mathematical discourse where students use sentence stems to critique reasoning, which would lead to deeper conceptual understanding. Additionally, creating opportunities for students to showcase their problem-solving approaches to the whole class would help validate diverse thinking strategies and reinforce a culture of learning from multiple perspectives."
            
            # Store human feedback
            human_update = self.supabase_db.store_human_feedback(
                output_id=output_id,
                score=human_score,
                feedback=human_feedback,
                improved_output=improved_output
            )
            
            self.assertTrue(human_update, "Failed to store human feedback")
            logger.info("Successfully stored human feedback")
        
        # Step 5: Run the backward pass with the evaluations
        logger.info("Step 5: Running backward pass")
        
        # Run the backward pass directly
        self.archer.run_backward_pass(evaluations)
        
        # Verify that we have candidate prompts
        self.assertIsNotNone(self.archer.candidate_prompts, "No candidate prompts generated")
        
        # If human validation was provided, we should have some candidate prompts
        if stored_output_ids:
            # Only assert if we actually stored output IDs
            self.assertGreater(len(self.archer.candidate_prompts), 0, "No candidate prompts generated")
            logger.info(f"Generated {len(self.archer.candidate_prompts)} candidate prompts")
        
        # Step 6: Test the optimized prompts with a new input
        logger.info("Step 6: Testing optimized prompts")
        
        # Try with a different component
        new_component_id = "3a"  # Communication
        new_input_data = {
            "input": SAMPLE_LOW_INFERENCE_NOTES,
            "component_id": new_component_id
        }
        
        # Run forward pass with the new input
        new_evaluations = self.archer.run_forward_pass(new_input_data)
        
        # Verify we got results
        self.assertIsNotNone(new_evaluations, "Second forward pass returned None")
        self.assertGreater(len(new_evaluations), 0, "No evaluations generated in second pass")
        
        # Check the content of the new evaluations
        for prompt, content, eval_result in new_evaluations:
            self.assertIsNotNone(content, "New generated content is None")
            self.assertGreater(len(content), 0, "New generated content is empty")
            self.assertIsNotNone(eval_result, "New evaluation result is None")
            self.assertIn("score", eval_result, "New evaluation missing score")
            
            logger.info(f"New generated content with score: {eval_result.get('score')}")
        
        logger.info("Direct Archer flow test completed successfully")
        
    def test_database_performance_metrics(self):
        """
        Test the database performance metrics functionality.
        """
        # Get performance metrics
        metrics = self.supabase_db.get_performance_metrics()
        
        # Basic validation of the metrics
        self.assertIsNotNone(metrics, "Performance metrics returned None")
        self.assertIsInstance(metrics, dict, "Performance metrics should be a dictionary")
        
        # Check for expected keys
        expected_keys = ["prompt_performance", "average_scores", "human_ai_agreement"]
        for key in expected_keys:
            self.assertIn(key, metrics, f"Performance metrics missing '{key}' key")
        
        logger.info("Database performance metrics test completed successfully")
        
    def test_best_prompts_retrieval(self):
        """
        Test the retrieval of the best prompts from the database.
        """
        # Get the current best prompts
        best_prompts = self.supabase_db.get_current_best_prompts(top_n=3)
        
        # This might return an empty list if no prompts have been evaluated yet
        self.assertIsNotNone(best_prompts, "Best prompts returned None")
        
        if best_prompts:
            for prompt in best_prompts:
                self.assertIsInstance(prompt, str, "Each best prompt should be a string")
                self.assertGreater(len(prompt), 0, "Best prompt is empty")
        
        logger.info("Best prompts retrieval test completed successfully")

    def tearDown(self):
        """
        Clean up any resources after the test.
        """
        # No specific cleanup needed as the database connection will be closed automatically
        pass


if __name__ == "__main__":
    unittest.main() 