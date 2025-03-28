import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import sys
import random
import json
from datetime import datetime
import uuid
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the parent directory is in the path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import after path is set
from data_labelling.archer.database.supabase import SupabaseDatabase
from archer.archer import Archer
from archer.helpers.prompt import Prompt
from archer.backwardPass.danielson_model import DanielsonModel
from eval.danielson import generate_single_component_evaluation, normalize_score_integer

# All Danielson components
DANIELSON_COMPONENTS = [
    "1a", "1b", "1c", "1d", "1e", "1f",
    "2a", "2b", "2c", "2d", "2e",
    "3a", "3b", "3c", "3d", "3e"
]

# Sample low-inference notes (you can replace with your actual data)
SAMPLE_LOW_INFERENCE_NOTES = [
    """The teacher begins class by welcoming students and displaying the day's agenda on the smart board. 
    Students enter and take their seats quietly. After taking attendance, the teacher reviews yesterday's homework 
    assignment on fractions. "Who can tell me how to add fractions with unlike denominators?" she asks. 
    Several students raise their hands, and she calls on a student in the front row. The student explains the process, 
    and the teacher affirms the answer while writing the steps on the whiteboard.""",
    
    """The classroom is arranged with desks in groups of four. There are anchor charts visible on the walls displaying 
    key vocabulary terms and mathematical concepts. The teacher moves between groups, checking progress and asking 
    probing questions. "What strategy are you using to solve this problem?" she asks one group. "How did you know to 
    apply that approach?" she asks another. Students are engaged in collaborative work, discussing their thinking and 
    writing in their math journals.""",
    
    """The teacher introduces the new lesson on linear equations. "Today we're going to learn how to graph lines using 
    slope-intercept form," she explains. The teacher models the process with two examples, asking students to identify 
    the slope and y-intercept in each equation. Several students appear confused, so the teacher provides an alternative 
    explanation using a real-world scenario about climbing stairs. After the demonstration, students begin working 
    independently on practice problems while the teacher circulates to provide assistance."""
]

class DanielsonArcherApp:
    """
    Gradio interface for the Archer-Danielson framework implementation.
    Provides a user-friendly interface for generating and validating Danielson 
    subcomponent summaries and visualizing optimization progress.
    """
    
    def __init__(self, 
                archer_instance: Optional[Archer] = None, 
                danielson_model: Optional[DanielsonModel] = None,
                supabase_db: Optional[SupabaseDatabase] = None,
                api_url: Optional[str] = None,
                api_key: Optional[str] = None,
                max_rounds: int = 2):
        
        """
        Initialize the DanielsonArcherApp instance.
        
        Args:
            archer_instance: Instance of the Archer class (optional)
            danielson_model: Instance of the DanielsonModel class (optional)
            supabase_db: Instance of the SupabaseDatabase class (optional)
            api_url: URL of the Argilla server (optional)
            api_key: API key for authentication (optional)
        """
        self.db = supabase_db or SupabaseDatabase(api_url=api_url, api_key=api_key)
        # Connect to the database and initialize datasets
        self.db.connect()
        self.db.initialize_datasets()
        
        # Initialize the Danielson model if not provided
        self.danielson_model = danielson_model or DanielsonModel(adalflow_enabled=True)
        
        # Initialize Archer if not provided
        if archer_instance is None:
            # Define initial prompts for Archer
            initial_prompts = [
                Prompt("Generate a comprehensive performance analysis and growth path for Danielson component {component_id} based on these low-inference notes: {input}"),
                Prompt("Create a detailed Danielson framework evaluation for component {component_id}. Analyze the teacher's performance using this evidence: {input}")
            ]
            
            # Store prompts in database and ensure they have proper IDs before initializing Archer
            if self.db:
                logger = logging.getLogger(__name__)
                logger.info("Storing initial prompts in database before Archer initialization")
                for i, prompt in enumerate(initial_prompts):
                    prompt_id = self.db.store_generator_prompt(content=prompt.content)
                    if prompt_id:
                        # Set the database ID on the prompt object
                        prompt.id = prompt_id
                        logger.info(f"Stored initial prompt {i+1} with ID: {prompt_id}")
                    else:
                        logger.warning(f"Failed to store initial prompt {i+1} in database")
            
            # Initialize Archer with the Danielson model
            self.archer = Archer(
                generator_model_name="gemini-2.0-flash",
                evaluator_model_name="gemini-2.0-flash",
                optimizer_model_name="gemini-2.0-flash",
                knowledge_base=["./data_labelling/eval"],  # Path to knowledge directories
                rubric=self._get_evaluation_rubric(),
                initial_prompts=initial_prompts,
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                human_validation_enabled=True,
                num_simulations_per_prompt=3,
                database_config={"api_url": api_url, "api_key": api_key}
            )
        else:
            self.archer = archer_instance
            
            # If we have an existing Archer instance, make sure its prompts have IDs
            if self.db and hasattr(self.archer, 'active_prompts') and self.archer.active_prompts:
                logger = logging.getLogger(__name__)
                logger.info("Ensuring existing Archer prompts have database IDs")
                for i, prompt in enumerate(self.archer.active_prompts):
                    if not hasattr(prompt, 'id') or not prompt.id:
                        prompt_id = self.db.store_generator_prompt(content=prompt.content)
                        if prompt_id:
                            prompt.id = prompt_id
                            logger.info(f"Stored existing prompt {i+1} with ID: {prompt_id}")
        
        # Initialize state variables
        self.current_data = None
        self.current_round = 1
        self.current_output_id = None
        self.app = None
        self.max_rounds = max_rounds

        
    def _get_evaluation_rubric(self) -> str:
        """
        Define the evaluation rubric for the Danielson framework summaries.
        
        Returns:
            str: The evaluation rubric
        """
        return """
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
    
    def generate_input_data(self) -> Tuple[str, str]:
        """
        Generate input data for the Danielson evaluation.
        This function selects a random low-inference note and Danielson component.
        
        Returns:
            Tuple[str, str]: A tuple of (low_inference_notes, component_id)
        """
        # Select a random low-inference note
        low_inference_notes = random.choice(SAMPLE_LOW_INFERENCE_NOTES)
        
        # Select a random Danielson component
        component_id = random.choice(DANIELSON_COMPONENTS)
        
        return low_inference_notes, component_id
    
    def generate_summary(self, low_inference_notes: str, component_id: str) -> Dict[str, Any]:
        """
        Generate a summary for the given component using the Danielson model.
        
        Args:
            low_inference_notes: The low-inference notes text
            component_id: The Danielson component ID (e.g. "1a")
            
        Returns:
            Dict[str, Any]: The generated summary and evaluation data
        """
        try:
            # Use the Danielson model to generate a component evaluation
            if self.archer:
                # Format the input for the Archer generator
                input_data = {
                    "input": low_inference_notes,
                    "component_id": component_id
                }
                
                # Run the forward pass to generate content
                evaluations = self.archer.run_forward_pass(input_data)
                
                if evaluations and len(evaluations) > 0:
                    # Extract content, evaluation and feedback from the first result
                    _, content, eval_result = evaluations[0]
                    
                    return {
                        "content": content,
                        "score": eval_result.get("score", 3),
                        "feedback": eval_result.get("feedback", ""),
                        "perfect_output": eval_result.get("improved_output", "")
                    }
            
            # Fallback to direct generation if Archer is not available
            result = generate_single_component_evaluation(low_inference_notes, component_id)
            
            if "error" in result:
                return {
                    "content": f"Error: {result['error']}",
                    "score": 1,
                    "feedback": "An error occurred during generation.",
                    "perfect_output": ""
                }
            
            return {
                "content": result.get("summary", ""),
                "score": normalize_score_integer(result.get("score", 3)),
                "feedback": "This was generated using the basic evaluation function.",
                "perfect_output": "A perfect output would include specific evidence from the notes and clear, actionable recommendations."
            }
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "content": f"Error: {str(e)}",
                "score": 1,
                "feedback": "An error occurred during generation.",
                "perfect_output": ""
            }
    
    def save_to_database(self, 
                        low_inference_notes: str, 
                        component_id: str,
                        content: str,
                        score: int,
                        feedback: str,
                        perfect_output: str) -> bool:
        """
        Save the evaluation data to the Argilla database.
        
        Args:
            low_inference_notes: The input low-inference notes
            component_id: The Danielson component ID
            content: The generated content/summary
            score: The evaluation score (1-5)
            feedback: The feedback text
            perfect_output: The perfect output example
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger = logging.getLogger(__name__)
            logger.info("Saving evaluation to database")
            
            # Format the input data
            input_data = json.dumps({
                "low_inference_notes": low_inference_notes,
                "component_id": component_id
            })
            
            # Get a valid prompt ID - either from active prompts or create a new one
            prompt_id = None
            
            # First try to get ID from Archer's active prompts
            if hasattr(self, 'archer') and self.archer and hasattr(self.archer, 'active_prompts') and self.archer.active_prompts:
                for prompt in self.archer.active_prompts:
                    if hasattr(prompt, 'id') and prompt.id:
                        # Verify the prompt ID exists in the database
                        try:
                            response = self.db.client.table("archer_prompts").select("id").eq("id", prompt.id).execute()
                            if response and hasattr(response, 'data') and len(response.data) > 0:
                                prompt_id = prompt.id
                                logger.info(f"Using verified prompt ID from active prompts: {prompt_id}")
                                break
                        except Exception as e:
                            logger.error(f"Error verifying prompt ID: {str(e)}")
            
            # If we couldn't get a valid ID, create a new prompt
            if not prompt_id:
                # Use the content of the first active prompt if available
                prompt_content = None
                if hasattr(self, 'archer') and self.archer and hasattr(self.archer, 'active_prompts') and self.archer.active_prompts:
                    prompt_content = self.archer.active_prompts[0].content
                else:
                    # Create a default prompt if no active prompts
                    prompt_content = f"Generate a comprehensive analysis for Danielson component {component_id} based on classroom observation notes."
                
                # Store the prompt in the database to get a valid ID
                prompt_id = self.db.store_generator_prompt(content=prompt_content)
                logger.info(f"Created new prompt in database with ID: {prompt_id}")
            
            if not prompt_id:
                logger.error("Failed to get a valid prompt ID")
                return False
                
            logger.info(f"Using prompt ID: {prompt_id}")
            
            # Store the generated content
            logger.info("Storing generated content")
            output_id = self.db.store_generated_content(
                input_data=input_data,
                content=content,
                prompt_id=prompt_id,
                round_num=self.current_round
            )
            
            if not output_id:
                logger.error("Failed to store generated content")
                return False
            
            # Store the output_id as an instance variable so it can be accessed in tests
            self.current_output_id = output_id
            
            logger.info(f"Generated content stored with output ID: {output_id}")
            
            # Store the evaluation
            logger.info("Storing evaluation")
            eval_success = self.db.store_evaluation(
                output_id=output_id,
                score=int(score),
                feedback=feedback,
                improved_output=perfect_output,
                is_human=True  # Mark this as human evaluation since it's coming from the UI
            )
            
            logger.info(f"Evaluation storage {'successful' if eval_success else 'failed'}")
            
            if eval_success:
                # Verify the evaluation was stored correctly by retrieving it
                logger.info("Verifying evaluation was stored correctly")
                try:
                    verified = self.db._get_latest_evaluation(output_id)
                    if verified:
                        logger.info("Evaluation successfully verified in database")
                        # Check if metadata is correct
                        metadata = verified.get("metadata", {})
                        if isinstance(metadata, dict):
                            is_human = metadata.get("is_human", "0")
                            logger.info(f"Evaluation is_human flag: {is_human}")
                            # Log all metadata keys for debugging
                            logger.info(f"Metadata keys found: {list(metadata.keys())}")
                        else:
                            logger.warning(f"Unexpected metadata structure: {type(metadata)}")
                    else:
                        logger.warning("Could not verify evaluation in database")
                except Exception as ve:
                    logger.error(f"Error verifying evaluation: {str(ve)}")
            
            return eval_success
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            return False
    
    def trigger_backward_pass(self) -> bool:
        """
        Trigger the backward pass in the Archer system.
        
        Returns:
            bool: True if backward pass is successful, False otherwise
        """
        logger = logging.getLogger(__name__)
        logger.info("Triggering backward pass")

        if self.current_round >= self.max_rounds:
            logger.info("Maximum number of rounds reached. Terminating backward pass optimization.")
            return False
        
        try:
            if self.archer is None:
                logger.warning("No Archer instance available. Skipping backward pass.")
                # Increment round number anyway for simulation
                self.current_round += 1
                return True
            
            # Get the evaluations from the database
            logger.info("Fetching validated evaluations from database")
            evaluations = self.db.get_validated_evaluations(limit=10)
            
            if evaluations is None or len(evaluations) == 0:
                logger.warning("No validated evaluations found for backward pass")
                
                # Check if there are any records at all in the database
                # Try to fetch any evaluations without the human filter
                try:
                    logger.info("Attempting to fetch any evaluations without human filter")
                    all_evaluations = self.db.datasets["evaluations"].records().to_list(flatten=True)
                    if all_evaluations:
                        logger.info(f"Found {len(all_evaluations)} total evaluations, but none are validated")
                        
                        # Check if there are evaluations that need to be manually validated
                        unvalidated = [e for e in all_evaluations 
                                      if e.get("metadata", {}).get("is_human", "0") == "0"]
                        
                        if unvalidated:
                            logger.info(f"Found {len(unvalidated)} unvalidated evaluations that need human validation")
                            logger.info("Please manually validate some evaluations before triggering backward pass")
                        else:
                            logger.info("No unvalidated evaluations found")
                    else:
                        logger.info("No evaluations found in the database")
                except Exception as e:
                    logger.error(f"Error checking for unvalidated evaluations: {str(e)}")
                
                self.current_round += 1
                return False
                
            logger.info(f"Retrieved {len(evaluations)} evaluations from database")
            
            # Transform database evaluations into the format Archer expects
            # Archer expects: [(Prompt, generated_content, evaluation_dict), ...]
            logger.info("Transforming evaluations for backward pass")
            archer_evaluations = []
            
            for _, evaluation in evaluations.iterrows():
                try:
                    # Extract data from evaluation
                    content = evaluation.get("generated_content", "")
                    
                    # Parse input data (should be JSON with component_id and input)
                    input_data = {}
                    try:
                        input_str = evaluation.get("input", "{}")
                        if isinstance(input_str, str) and (input_str.startswith('{') or input_str.startswith('[')):
                            input_data = json.loads(input_str)
                        else:
                            input_data = {"input": input_str}
                    except Exception as e:
                        logger.error(f"Error parsing input data: {str(e)}")
                        input_data = {"component_id": "1a", "input": input_str}
                    
                    # Create a prompt object from one of the current active prompts or from the database
                    prompt_id = evaluation.get("prompt_id", "")
                    prompt_obj = None
                    
                    # Try to find the matching prompt by ID
                    if hasattr(self.archer, 'active_prompts'):
                        for p in self.archer.active_prompts:
                            if getattr(p, 'id', None) == prompt_id:
                                prompt_obj = p
                                break
                    
                    # If not found, use the first active prompt or create a default one
                    if prompt_obj is None:
                        if hasattr(self.archer, 'active_prompts') and self.archer.active_prompts:
                            prompt_obj = self.archer.active_prompts[0]
                        elif hasattr(self.archer, 'active_generator_prompts') and self.archer.active_generator_prompts:
                            prompt_obj = self.archer.active_generator_prompts[0]
                        else:
                            # Create a default prompt if none available
                            from data_labelling.archer.helpers.prompt import Prompt
                            prompt_obj = Prompt("Generate an evaluation for component {component_id} based on {input}")
                    
                    # Get score with safe default
                    score_value = evaluation.get("score")
                    if score_value is None:
                        score = 3.0  # Default score if missing
                    else:
                        try:
                            score = float(score_value)
                        except (ValueError, TypeError):
                            score = 3.0  # Default on conversion error
                    
                    # Create evaluation dict
                    eval_dict = {
                        "score": score,
                        "feedback": evaluation.get("feedback", ""),
                        "improved_output": evaluation.get("improved_output", "")
                    }
                    
                    # Add to archer evaluations
                    archer_evaluations.append((prompt_obj, content, eval_dict))
                    logger.debug(f"Added evaluation: score={eval_dict['score']}")
                except Exception as e:
                    logger.error(f"Error processing evaluation: {str(e)}")
            
            if not archer_evaluations:
                logger.warning("No valid evaluations to process after transformation")
                self.current_round += 1
                return False
            
            # Trigger the backward pass in the Archer system
            logger.info(f"Calling Archer backward pass with {len(archer_evaluations)} evaluations")
            logger.debug("First evaluation prompt: " + archer_evaluations[0][0].content[:50] + "...")
            logger.debug("First evaluation score: " + str(archer_evaluations[0][2].get("score", "N/A")))
            
            # Call the backward pass
            self.archer.run_backward_pass(archer_evaluations)
            
            # Increment the round number
            self.current_round += 1
            
            logger.info(f"Backward pass completed. New round: {self.current_round}")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering backward pass: {str(e)}", exc_info=True)
            return False
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface components.
        
        Returns:
            gr.Blocks: The configured Gradio interface
        """
        with gr.Blocks(title="Danielson Evaluation System") as app:
            gr.Markdown("# Danielson Framework Evaluation System")
            gr.Markdown("Generate and evaluate Danielson component evaluations with AI assistance.")
            
            with gr.Tab("Generate & Evaluate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        low_inference_notes = gr.Textbox(
                            label="Low Inference Notes",
                            placeholder="Enter classroom observation notes here...",
                            lines=8
                        )
                        component_id = gr.Dropdown(
                            label="Danielson Component",
                            choices=DANIELSON_COMPONENTS,
                            value="1a"
                        )
                        gen_input_btn = gr.Button("Generate Sample Input")
                        gen_btn = gr.Button("Generate Summary", variant="primary")
                    
                    with gr.Column(scale=3):
                        generated_content = gr.Textbox(
                            label="Generated Evaluation",
                            placeholder="AI-generated evaluation will appear here...",
                            lines=10
                        )
                        score = gr.Slider(
                            label="Quality Score (1-5)",
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=3
                        )
                        feedback = gr.Textbox(
                            label="Feedback",
                            placeholder="Provide feedback on the quality of the generated evaluation...",
                            lines=3
                        )
                        perfect_output = gr.Textbox(
                            label="Perfect Output Example",
                            placeholder="Example of ideal output will appear here after generation...",
                            lines=5
                        )
                        save_btn = gr.Button("Save Evaluation", variant="primary")
                        
                # Add a section to display the current prompts
                with gr.Accordion("Current Active Prompts", open=False):
                    current_prompts = gr.Textbox(
                        label="Active Prompts",
                        placeholder="The currently active prompts will be displayed here.",
                        lines=5,
                        interactive=False
                    )
                    update_prompts_btn = gr.Button("Show Current Prompts")
                        
                with gr.Row():
                    status_msg = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tab("Optimization"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Prompt Optimization")
                        gr.Markdown(
                            "Click the button below to trigger the backward pass and optimize prompts based on collected feedback."
                        )
                        optimize_btn = gr.Button("Optimize Prompts", variant="primary")
                        optimization_status = gr.Textbox(label="Optimization Status", interactive=False)
                        
                        # Display the current round
                        current_round_display = gr.Textbox(
                            label="Current Round",
                            value=f"Round {self.current_round}",
                            interactive=False
                        )
                
                with gr.Accordion("Optimized Prompts", open=False):
                    optimized_prompts = gr.Textbox(
                        label="Prompts After Optimization",
                        placeholder="The optimized prompts will be displayed here after optimization.",
                        lines=8,
                        interactive=False
                    )
            
            # Function to generate input data
            def on_generate_input():
                low_inf_notes, comp_id = self.generate_input_data()
                return low_inf_notes, comp_id
            
            # Function to generate summary
            def on_generate_summary(notes, comp):
                result = self.generate_summary(notes, comp)
                return result["content"], result["score"], result["feedback"], result["perfect_output"]
            
            # Function to save evaluation
            def on_save(notes, comp, content, score, fb, perfect):
                success = self.save_to_database(notes, comp, content, score, fb, perfect)
                if success:
                    return "Evaluation saved successfully!"
                else:
                    return "Error saving evaluation. Please try again."
            
            # Function to optimize prompts
            def on_optimize():
                success = self.trigger_backward_pass()
                if success:
                    # Update the current round display
                    current_round = f"Round {self.current_round}"
                    # Get the optimized prompts for display
                    prompt_text = self._get_current_prompts_text()
                    return "Optimization complete! Prompts have been updated for the next round.", current_round, prompt_text
                else:
                    return "Error during optimization. Please check logs.", f"Round {self.current_round}", ""
            
            # Function to display current prompts
            def on_show_prompts():
                return self._get_current_prompts_text()
                
            # Connect UI components to functions
            gen_input_btn.click(on_generate_input, outputs=[low_inference_notes, component_id])
            gen_btn.click(on_generate_summary, inputs=[low_inference_notes, component_id], 
                         outputs=[generated_content, score, feedback, perfect_output])
            save_btn.click(on_save, inputs=[low_inference_notes, component_id, generated_content, 
                                          score, feedback, perfect_output], 
                          outputs=[status_msg])
            optimize_btn.click(on_optimize, outputs=[optimization_status, current_round_display, optimized_prompts])
            update_prompts_btn.click(on_show_prompts, outputs=[current_prompts])
            
        self.app = app
        return app
    
    def _get_current_prompts_text(self) -> str:
        """
        Get the text representation of the current active prompts.
        
        Returns:
            str: The formatted text of the current prompts
        """
        if not self.archer:
            logger.warning("No Archer instance available for retrieving prompts")
            return "No Archer instance available."
            
        try:
            # Get the active prompts from the Archer instance
            # First try active_prompts, then fall back to active_generator_prompts
            if hasattr(self.archer, 'active_prompts'):
                active_prompts = self.archer.active_prompts
            elif hasattr(self.archer, 'active_generator_prompts'):
                active_prompts = self.archer.active_generator_prompts
            else:
                logger.warning("No active prompts attribute found in Archer instance")
                return "No active prompts found."
                
            if not active_prompts:
                logger.warning("No active prompts found in Archer instance")
                return "No active prompts found."
                
            # Format the prompts for display
            prompt_text = "CURRENT ACTIVE PROMPTS:\n\n"
            for i, prompt in enumerate(active_prompts):
                prompt_text += f"Prompt {i+1}:\n{prompt.content}\n\n"
                
            logger.info(f"Retrieved {len(active_prompts)} active prompts for display")
            return prompt_text
        except Exception as e:
            logger.error(f"Error getting current prompts: {str(e)}")
            return f"Error retrieving prompts: {str(e)}"
    
    def launch(self, share: bool = False, server_port: Optional[int] = None):
        """
        Launch the Gradio app.
        
        Args:
            share: Whether to create a public link for the interface
            server_port: The port to run the server on
        """
        if self.app is None:
            self.create_interface()
        
        self.app.launch(share=share, server_port=server_port)

    def fix_missing_prompt_ids(self, limit: int = 100) -> int:
        """
        Fix evaluations with missing prompt IDs.
        
        Args:
            limit: Maximum number of evaluations to fix
            
        Returns:
            Number of fixed evaluations
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Fixing missing prompt IDs (limit: {limit})")
        
        if not self.db:
            logger.error("No database connection available")
            return 0
            
        try:
            fixed_count = self.db.fix_missing_prompt_ids(limit=limit)
            logger.info(f"Fixed {fixed_count} evaluations with missing prompt IDs")
            return fixed_count
        except Exception as e:
            logger.error(f"Error fixing missing prompt IDs: {str(e)}")
            return 0


# Run the app if this file is executed directly
if __name__ == "__main__":
    app = DanielsonArcherApp()
    app.launch()