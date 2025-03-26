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
from archer.database.argilla import ArgillaDatabase
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
                argilla_db: Optional[ArgillaDatabase] = None,
                api_url: Optional[str] = None,
                api_key: Optional[str] = None):
        """
        Initialize the DanielsonArcherApp instance.
        
        Args:
            archer_instance: Instance of the Archer class (optional)
            danielson_model: Instance of the DanielsonModel class (optional)
            argilla_db: Instance of the ArgillaDatabase class (optional)
            api_url: URL of the Argilla server (optional)
            api_key: API key for authentication (optional)
        """
        self.db = argilla_db or ArgillaDatabase(api_url=api_url, api_key=api_key)
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
        
        # Initialize state variables
        self.current_data = None
        self.current_round = 1
        self.app = None
        
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
            # Format the input data
            input_data = json.dumps({
                "low_inference_notes": low_inference_notes,
                "component_id": component_id
            })
            
            # Generate a prompt ID if we don't have one from Archer
            prompt_id = str(uuid.uuid4())
            
            # Store the generated content
            output_id = self.db.store_generated_content(
                input_data=input_data,
                content=content,
                prompt_id=prompt_id,
                round_num=self.current_round
            )
            
            if not output_id:
                logger.error("Failed to store generated content")
                return False
            
            # Store the evaluation
            eval_success = self.db.store_evaluation(
                output_id=output_id,
                score=int(score),
                feedback=feedback,
                improved_output=perfect_output,
                is_human=True  # Mark this as human evaluation since it's coming from the UI
            )
            
            return eval_success
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            return False
    
    def trigger_optimization(self) -> str:
        """
        Trigger the optimization process in Archer (backward pass).
        
        Returns:
            str: Status message
        """
        try:
            if not self.archer:
                self.current_round += 1
                return f"No Archer instance available. Simulated optimization. Advanced to round {self.current_round}."
            
            # Run backward pass
            evaluations = []  # This would normally come from the database
            self.archer.run_backward_pass(evaluations)
            
            # Increment round number
            self.current_round += 1
            
            return f"Optimization completed. Advanced to round {self.current_round}."
        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}")
            return f"Error in optimization: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface.
        
        Returns:
            gr.Blocks: The Gradio Blocks interface
        """
        with gr.Blocks(title="Danielson Framework - Archer Optimization") as app:
            gr.Markdown("# Danielson Framework Component Summary Optimization")
            gr.Markdown("This application uses Archer to optimize prompts for generating Danielson framework component summaries.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    low_inference_notes = gr.Textbox(
                        label="Low Inference Notes",
                        placeholder="Enter or generate low inference notes...",
                        lines=10
                    )
                    
                    component_id = gr.Dropdown(
                        label="Danielson Component",
                        choices=DANIELSON_COMPONENTS,
                        value="1a"
                    )
                    
                    with gr.Row():
                        generate_input_btn = gr.Button("Generate Random Input")
                        generate_summary_btn = gr.Button("Generate Summary", variant="primary")
                
                with gr.Column(scale=3):
                    summary_output = gr.Textbox(
                        label="Generated Summary",
                        placeholder="The generated summary will appear here...",
                        lines=15
                    )
                    
                    with gr.Row():
                        score = gr.Slider(
                            label="Quality Score",
                            minimum=1,
                            maximum=5,
                            step=1,
                            value=3
                        )
                    
                    feedback = gr.Textbox(
                        label="Feedback",
                        placeholder="Provide feedback on how to improve the summary...",
                        lines=5
                    )
                    
                    perfect_output = gr.Textbox(
                        label="Perfect Output Example",
                        placeholder="Provide an example of what a perfect summary would look like...",
                        lines=10
                    )
                    
                    with gr.Row():
                        save_btn = gr.Button("Save & Get New Data")
                        optimize_btn = gr.Button("Run Optimization", variant="secondary")
            
            with gr.Row():
                status_output = gr.Textbox(label="Status", lines=2)
            
            # Event handlers
            def on_generate_input():
                notes, comp = self.generate_input_data()
                return [notes, comp]
            
            def on_generate_summary(notes, comp):
                result = self.generate_summary(notes, comp)
                return [
                    result["content"],
                    result["score"],
                    result["feedback"],
                    result["perfect_output"]
                ]
            
            def on_save(notes, comp, content, score, fb, perfect):
                success = self.save_to_database(notes, comp, content, score, fb, perfect)
                if success:
                    new_notes, new_comp = self.generate_input_data()
                    new_result = self.generate_summary(new_notes, new_comp)
                    return [
                        new_notes,
                        new_comp,
                        new_result["content"],
                        new_result["score"],
                        new_result["feedback"],
                        new_result["perfect_output"],
                        "Data saved successfully. Generated new data."
                    ]
                else:
                    return [
                        notes, comp, content, score, fb, perfect,
                        "Failed to save data. Please try again."
                    ]
            
            def on_optimize():
                status = self.trigger_optimization()
                return status
            
            # Connect events
            generate_input_btn.click(
                on_generate_input,
                inputs=[],
                outputs=[low_inference_notes, component_id]
            )
            
            generate_summary_btn.click(
                on_generate_summary,
                inputs=[low_inference_notes, component_id],
                outputs=[summary_output, score, feedback, perfect_output]
            )
            
            save_btn.click(
                on_save,
                inputs=[
                    low_inference_notes, component_id, 
                    summary_output, score, feedback, perfect_output
                ],
                outputs=[
                    low_inference_notes, component_id, 
                    summary_output, score, feedback, perfect_output,
                    status_output
                ]
            )
            
            optimize_btn.click(
                on_optimize,
                inputs=[],
                outputs=[status_output]
            )
            
            self.app = app
            return app
    
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


# Run the app if this file is executed directly
if __name__ == "__main__":
    app = DanielsonArcherApp()
    app.launch()