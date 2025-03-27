import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import sys
from datetime import datetime, timedelta
import uuid
import logging
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import after path is set
from archer.database.argilla import ArgillaDatabase
from archer.helpers.prompt import Prompt

# Track optimization status
_last_backward_pass_attempt = datetime.now()
_backward_pass_min_interval = timedelta(minutes=5)  # Minimum time between backward pass attempts
_failed_attempts = 0
_max_consecutive_failures = 3
_failure_backoff_multiplier = 2

class GradioApp:
    """
    Gradio interface for human validation and visualization of Archer system.
    Provides a user-friendly interface for reviewing and modifying AI evaluations,
    and visualizing system performance.
    """
    
    def __init__(self, archer_instance: Optional[Any] = None, 
                argilla_db: Optional[ArgillaDatabase] = None,
                api_url: Optional[str] = None,
                api_key: Optional[str] = None):
        """
        Initialize the GradioApp instance.
        
        Args:
            archer_instance: Instance of the Archer class (optional)
            argilla_db: Instance of the ArgillaDatabase class (optional)
            api_url: URL of the Argilla server (optional)
            api_key: API key for authentication (optional)
        """
        self.archer = archer_instance
        # Create a new ArgillaDatabase instance if one wasn't provided
        self.db = argilla_db or ArgillaDatabase(api_url=api_url, api_key=api_key)
        # Connect to the database and initialize datasets
        self.db.connect()
        self.db.initialize_datasets()
        # Initialize state variables
        self.current_data = None
        self.current_round = 1
        self.app = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data for annotation from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing the data for annotation
        """
        try:
            logger.info(f"Loading data for round {self.current_round}")
            df = self.db.get_current_data_for_annotation(self.current_round)
            
            if df is None or df.empty:
                logger.warning("No data found. Creating empty DataFrame with correct structure.")
                df = pd.DataFrame(columns=[
                    "output_id", "input", "eval_content", "eval_score", 
                    "eval_feedback", "eval_perfect_output", "prompt_id"
                ])
            
            # Store the current data
            self.current_data = df
            
            logger.info(f"Loaded {len(df)} records for annotation")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=[
                "output_id", "input", "eval_content", "eval_score", 
                "eval_feedback", "eval_perfect_output", "prompt_id"
            ])
    
    def save_data(self, dataframe: pd.DataFrame) -> bool:
        """
        Save annotated data to the database.
        
        Args:
            dataframe: DataFrame containing the annotated data
            
        Returns:
            bool: True if save is successful, False otherwise
        """
        try:
            logger.info(f"Saving {len(dataframe)} records to database")
            
            success = True
            for _, row in dataframe.iterrows():
                # Store human feedback
                feedback_success = self.db.store_human_feedback(
                    output_id=row['output_id'],
                    score=int(row['eval_score']),
                    feedback=row['eval_feedback'],
                    improved_output=row['eval_perfect_output']
                )
                
                if not feedback_success:
                    logger.warning(f"Failed to save feedback for output {row['output_id']}")
                    success = False
            
            logger.info("Data saved successfully" if success else "Some records failed to save")
            return success
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def trigger_backward_pass(self) -> dict:
        """
        Trigger the backward pass optimization process.
        
        This function checks if there is an Archer instance available, retrieves validated
        evaluations from the database, and triggers the backward pass optimization if 
        evaluations are found.
        
        Returns:
            dict: A status message indicating whether the backward pass was triggered.
        """
        global _last_backward_pass_attempt, _failed_attempts
        
        logger = logging.getLogger(__name__)
        logger.info("Triggering backward pass")
        
        # Check if enough time has passed since the last attempt
        time_since_last_attempt = datetime.now() - _last_backward_pass_attempt
        required_interval = _backward_pass_min_interval * (_failure_backoff_multiplier ** min(_failed_attempts, 5))
        
        if time_since_last_attempt < required_interval:
            remaining_seconds = (required_interval - time_since_last_attempt).total_seconds()
            logger.warning(
                f"Attempt to trigger backward pass too soon after previous attempt. "
                f"Please wait {remaining_seconds:.0f} seconds before retrying."
            )
            return {
                "status": "error", 
                "message": f"Backward pass was attempted too recently. Please wait {remaining_seconds:.0f} seconds before retrying."
            }
        
        # Update the last attempt time
        _last_backward_pass_attempt = datetime.now()
        
        # Check if there is an Archer instance
        if self.archer is None:
            logger.error("No Archer instance available for backward pass")
            return {"status": "error", "message": "No Archer instance available"}
        
        # Get validated evaluations from the database
        try:
            logger.info("Fetching validated evaluations from database")
            evaluations = self.db.get_validated_evaluations()
            logger.info(f"Found {len(evaluations) if evaluations else 0} validated evaluations")
            
            if not evaluations:
                logger.warning("No evaluations found, skipping backward pass")
                return {"status": "warning", "message": "No evaluations found for optimization"}
            
            # Transform evaluations to the format expected by Archer
            transformed_evaluations = []
            for eval_data in evaluations:
                try:
                    prompt_content = eval_data.get('prompt', '')
                    prompt = Prompt(
                        content=prompt_content,
                        score=eval_data.get('score', 3.0),
                        feedback_or_generation=eval_data.get('feedback', '')
                    )
                    transformed_evaluations.append(
                        (prompt, eval_data.get('generated_text', ''), eval_data)
                    )
                except Exception as e:
                    logger.error(f"Error transforming evaluation: {str(e)}")
                    # Continue with other evaluations
            
            if not transformed_evaluations:
                logger.error("Failed to transform any evaluations, skipping backward pass")
                _failed_attempts += 1
                return {"status": "error", "message": "Failed to transform evaluations"}
            
            # Run the backward pass
            logger.info(f"Running backward pass with {len(transformed_evaluations)} evaluations")
            start_time = time.time()
            
            # Track success/failure
            try:
                success = self.archer.run_backward_pass(transformed_evaluations)
                execution_time = time.time() - start_time
                
                if success:
                    logger.info(f"Backward pass completed successfully in {execution_time:.2f} seconds")
                    # Reset failure count on success
                    _failed_attempts = 0
                    return {"status": "success", "message": "Backward pass completed successfully"}
                else:
                    logger.warning(f"Backward pass failed after {execution_time:.2f} seconds")
                    _failed_attempts += 1
                    # Increase backoff time for consecutive failures
                    if _failed_attempts >= _max_consecutive_failures:
                        logger.error(
                            f"Backward pass has failed {_failed_attempts} consecutive times. "
                            f"Increasing backoff interval significantly."
                        )
                    return {
                        "status": "error",
                        "message": f"Backward pass failed (attempt {_failed_attempts})"
                    }
                    
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Error during backward pass: {str(e)}")
                _failed_attempts += 1
                return {"status": "error", "message": f"Error during backward pass: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error fetching evaluations: {str(e)}")
            _failed_attempts += 1
            return {"status": "error", "message": f"Error fetching evaluations: {str(e)}"}
    
    def create_prompt_performance_chart(self) -> plt.Figure:
        """
        Create a chart showing prompt performance across rounds.
        
        Returns:
            plt.Figure: Matplotlib figure containing the chart
        """
        try:
            logger.info("Creating prompt performance chart")
            
            # Get performance metrics from the database
            metrics = self.db.get_performance_metrics()
            
            # Create figure
            fig = plt.figure(figsize=(10, 6))
            
            # Extract prompt data
            prompts = metrics.get("prompts", [])
            
            if not prompts:
                plt.title("No prompt data available")
                return fig
            
            # Group prompts by generation
            generations = {}
            for prompt in prompts:
                gen = prompt["generation"]
                if gen not in generations:
                    generations[gen] = []
                generations[gen].append(prompt)
            
            # Get unique prompt IDs and generations
            unique_gens = sorted(generations.keys())
            
            if not unique_gens:
                plt.title("No generation data available")
                return fig
            
            # Set positions for bars
            bar_width = 0.8 / len(unique_gens)
            positions = np.arange(min(5, max(len(generations[gen]) for gen in unique_gens)))
            
            # Create bars for each generation
            for i, gen in enumerate(unique_gens):
                gen_prompts = generations[gen][:len(positions)]
                gen_scores = [p["avg_score"] for p in gen_prompts]
                
                # Pad with zeros if needed
                while len(gen_scores) < len(positions):
                    gen_scores.append(0)
                
                # Plot bars
                plt.bar(
                    positions + i * bar_width - (len(unique_gens) - 1) * bar_width / 2, 
                    gen_scores[:len(positions)], 
                    bar_width, 
                    label=f'Generation {gen}'
                )
            
            # Add labels and title
            plt.xlabel('Prompt Index')
            plt.ylabel('Average Score')
            plt.title('Prompt Performance Across Generations')
            plt.xticks(positions, [f'Prompt {i+1}' for i in positions])
            plt.legend()
            plt.ylim(0, 5.5)
            
            logger.info("Prompt performance chart created")
            return fig
        except Exception as e:
            logger.error(f"Error creating prompt performance chart: {str(e)}")
            # Return empty figure
            return plt.figure()
    
    def create_model_improvement_chart(self) -> plt.Figure:
        """
        Create a chart showing model improvement over time.
        
        Returns:
            plt.Figure: Matplotlib figure containing the chart
        """
        try:
            logger.info("Creating model improvement chart")
            
            # Get performance metrics from the database
            metrics = self.db.get_performance_metrics()
            
            # Create figure
            fig = plt.figure(figsize=(10, 6))
            
            # Extract score data
            scores = metrics.get("scores", [])
            rounds = metrics.get("rounds", [])
            moving_avg = metrics.get("moving_avg", [])
            
            if not scores or not rounds:
                plt.title("No score data available")
                return fig
            
            # Create scatter plot of individual scores
            plt.scatter(rounds, scores, alpha=0.5, label='Individual Scores')
            
            # Create line plot of moving average
            if moving_avg:
                # Calculate x-coordinates for moving average (centered)
                window_size = min(5, len(scores))
                ma_x = rounds[window_size//2:-(window_size//2)] if window_size > 1 else rounds
                if len(ma_x) > len(moving_avg):
                    ma_x = ma_x[:len(moving_avg)]
                elif len(ma_x) < len(moving_avg):
                    moving_avg = moving_avg[:len(ma_x)]
                
                plt.plot(ma_x, moving_avg, 'r-', linewidth=2, label=f'{window_size}-Point Moving Average')
            
            # Add regression line
            if len(rounds) > 1:
                z = np.polyfit(rounds, scores, 1)
                p = np.poly1d(z)
                plt.plot(sorted(rounds), p(sorted(rounds)), "b--", linewidth=1, label='Trend Line')
            
            # Add labels and title
            plt.xlabel('Round')
            plt.ylabel('Score')
            plt.title('Model Performance Over Time')
            plt.legend()
            plt.ylim(0, 5.5)
            plt.grid(True, alpha=0.3)
            
            logger.info("Model improvement chart created")
            return fig
        except Exception as e:
            logger.error(f"Error creating model improvement chart: {str(e)}")
            # Return empty figure
            return plt.figure()
    
    def create_prompt_maintenance_chart(self) -> plt.Figure:
        """
        Create a chart showing prompt maintenance/survivorship.
        
        Returns:
            plt.Figure: Matplotlib figure containing the chart
        """
        try:
            logger.info("Creating prompt maintenance chart")
            
            # Get performance metrics from the database
            metrics = self.db.get_performance_metrics()
            
            # Create figure
            fig = plt.figure(figsize=(10, 6))
            
            # Extract survivorship data
            survivorship = metrics.get("prompt_survivorship", {})
            
            if not survivorship:
                plt.title("No survivorship data available")
                return fig
            
            # Extract prompt IDs and their generation spans
            prompt_ids = list(survivorship.keys())
            
            if not prompt_ids:
                plt.title("No prompt data available")
                return fig
            
            # Sort prompts by their first appearance
            prompt_ids.sort(key=lambda pid: min(survivorship[pid]["generations"]))
            
            # Plot each prompt's journey
            for i, pid in enumerate(prompt_ids[:10]):  # Show at most 10 prompts
                generations = survivorship[pid]["generations"]
                scores = survivorship[pid]["scores"]
                
                # Plot the journey line
                plt.plot(generations, [i] * len(generations), 'b-', linewidth=2)
                
                # Plot points with score-based size
                sizes = [max(5, s * 20) for s in scores]
                plt.scatter(generations, [i] * len(generations), s=sizes, c='blue', alpha=0.7)
                
                # Add label
                plt.text(min(generations) - 0.2, i, f'Prompt {pid[:5]}...', 
                        ha='right', va='center', fontsize=9)
            
            # Add labels and title
            plt.xlabel('Generation')
            plt.ylabel('Prompt')
            plt.title('Prompt Survivorship Across Generations')
            plt.yticks([])
            plt.grid(True, axis='x', alpha=0.3)
            
            # Add score legend
            score_examples = [1, 2, 3, 4, 5]
            for i, score in enumerate(score_examples):
                size = max(5, score * 20)
                plt.scatter([i+1], [-1], s=size, c='blue', alpha=0.7)
                plt.text(i+1, -1.5, f'Score {score}', ha='center', va='center', fontsize=8)
            
            logger.info("Prompt maintenance chart created")
            return fig
        except Exception as e:
            logger.error(f"Error creating prompt maintenance chart: {str(e)}")
            # Return empty figure
            return plt.figure()
    
    def create_gradio_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface with all components.
        
        Returns:
            gr.Blocks: Gradio Blocks interface
        """
        logger.info("Creating Gradio interface")
        
        # Load initial data
        initial_data = self.load_data()
        
        # Create interface
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("# Archer: Human Validation Interface")
            gr.Markdown("Review and edit AI-generated evaluations in the table below.")
            
            # Store state
            current_df = gr.State(initial_data)
            
            # Create tabs for the main interface
            with gr.Tabs():
                # Tab 1: Data Annotation
                with gr.TabItem("Data Annotation"):
                    # Round number display
                    round_display = gr.Markdown(f"## Current Round: {self.current_round}")
                    
                    # Create the editable dataframe component
                    visible_cols = ["input", "eval_content", "eval_score", "eval_feedback", "eval_perfect_output"]
                    all_cols = ["output_id", "input", "eval_content", "eval_score", "eval_feedback", "eval_perfect_output", "prompt_id"]
                    
                    # Extract the data to display
                    display_data = initial_data[visible_cols] if not initial_data.empty else pd.DataFrame(columns=visible_cols)
                    
                    dataframe = gr.Dataframe(
                        value=display_data,
                        headers=visible_cols,
                        datatype=["str", "str", "number", "str", "str"],
                        col_count=(len(visible_cols), "fixed"),
                        interactive=[False, True, True, True, True],  # First column (input) is not editable
                        wrap=True,
                        height=500,
                        max_rows=20
                    )
                    
                    # Status message for saves
                    status_msg = gr.Textbox(label="Status", value="Ready", interactive=False)
                    
                    # Buttons row
                    with gr.Row():
                        # Save button
                        save_btn = gr.Button("Save Current Data")
                        
                        # Refresh button to trigger the backward pass
                        refresh_btn = gr.Button("REFRESH (Save & Get New Data)", variant="primary")
                
                # Tab 2: Performance Tracking
                with gr.TabItem("Performance Tracking"):
                    # Button for updating visualizations
                    update_viz_btn = gr.Button("Update Visualizations")
                    
                    # Create placeholders for visualization components
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Prompt Performance Across Generations")
                            prompt_perf_plot = gr.Plot()
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Model Improvement Over Time")
                            model_improve_plot = gr.Plot()
                        
                        with gr.Column():
                            gr.Markdown("### Prompt Maintenance Tracking")
                            prompt_maintain_plot = gr.Plot()
            
            # Define event handlers
            
            # Function to convert UI DataFrame back to full DataFrame with hidden columns
            def update_full_df(ui_df, full_df):
                try:
                    if ui_df is None or full_df is None or ui_df.empty or full_df.empty:
                        return full_df
                    
                    # Copy the editable columns from ui_df to full_df
                    updated_df = full_df.copy()
                    for col in ["eval_content", "eval_score", "eval_feedback", "eval_perfect_output"]:
                        if col in ui_df.columns and col in updated_df.columns:
                            updated_df[col] = ui_df[col].values
                    
                    return updated_df
                except Exception as e:
                    logger.error(f"Error updating full DataFrame: {str(e)}")
                    return full_df
            
            # When save is clicked, update the full DataFrame and save to database
            def on_save_click(ui_df, full_df):
                try:
                    # Update full DataFrame with UI values
                    updated_df = update_full_df(ui_df, full_df)
                    
                    # Save to database
                    success = self.save_data(updated_df)
                    
                    # Update state
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    return updated_df, f"Last saved: {timestamp}" if success else "Error saving data"
                except Exception as e:
                    logger.error(f"Error in save operation: {str(e)}")
                    return full_df, f"Error: {str(e)}"
            
            # When refresh is clicked, save data, trigger backward pass, and load new data
            def on_refresh_click(ui_df, full_df):
                try:
                    # First update and save the current data
                    updated_df = update_full_df(ui_df, full_df)
                    save_success = self.save_data(updated_df)
                    
                    if not save_success:
                        return ui_df, full_df, "Error saving data", round_display.value
                    
                    # Trigger the backward pass
                    backward_status = self.trigger_backward_pass()
                    
                    if backward_status["status"] == "error":
                        return ui_df, full_df, backward_status["message"], round_display.value
                    
                    # Load new data
                    new_df = self.load_data()
                    
                    # Prepare visible data for UI
                    visible_cols = ["input", "eval_content", "eval_score", "eval_feedback", "eval_perfect_output"]
                    new_ui_df = new_df[visible_cols] if not new_df.empty else pd.DataFrame(columns=visible_cols)
                    
                    # Update round display
                    new_round_display = f"## Current Round: {self.current_round}"
                    
                    return new_ui_df, new_df, "Data refreshed successfully!", new_round_display
                except Exception as e:
                    logger.error(f"Error in refresh operation: {str(e)}")
                    return ui_df, full_df, f"Error: {str(e)}", round_display.value
            
            # When update visualizations is clicked, regenerate all plots
            def on_update_viz_click():
                try:
                    prompt_perf = self.create_prompt_performance_chart()
                    model_improve = self.create_model_improvement_chart()
                    prompt_maintain = self.create_prompt_maintenance_chart()
                    
                    return prompt_perf, model_improve, prompt_maintain
                except Exception as e:
                    logger.error(f"Error updating visualizations: {str(e)}")
                    # Return empty figures
                    return plt.figure(), plt.figure(), plt.figure()
            
            # Connect the buttons to the functions
            save_btn.click(
                fn=on_save_click,
                inputs=[dataframe, current_df],
                outputs=[current_df, status_msg]
            )
            
            refresh_btn.click(
                fn=on_refresh_click,
                inputs=[dataframe, current_df],
                outputs=[dataframe, current_df, status_msg, round_display]
            )
            
            update_viz_btn.click(
                fn=on_update_viz_click,
                inputs=[],
                outputs=[prompt_perf_plot, model_improve_plot, prompt_maintain_plot]
            )
            
            # Also update the visualizations on load
            app.load(
                fn=on_update_viz_click,
                inputs=[],
                outputs=[prompt_perf_plot, model_improve_plot, prompt_maintain_plot]
            )
        
        logger.info("Gradio interface created successfully")
        return app
    
    def launch(self, share: bool = False, server_port: Optional[int] = None):
        """
        Launch the Gradio app.
        
        Args:
            share: Whether to create a publicly shareable link
            server_port: Port to run the server on
        """
        logger.info("Launching Gradio app")
        self.app = self.create_gradio_interface()
        self.app.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    # If this file is run directly, create and launch the app
    app = GradioApp()
    app.launch() 