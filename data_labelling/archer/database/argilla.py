import argilla as rg
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArgillaDatabase:
    """
    Handles all interactions with the Argilla database for the Archer system.
    
    This class implements the revised schema:
    - Records: Main table with all generated content and evaluations
    - Generator Prompts: Stores generator prompts with metrics
    - Evaluator Prompts: Stores evaluator prompts
    - Rounds: Tracks iteration information
    - Prompt Lineage: Tracks prompt evolution
    """
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the ArgillaDatabase instance.
        
        Args:
            api_url: URL of the Argilla server (optional, defaults to environment variable)
            api_key: API key for authentication (optional, defaults to environment variable)
        """
        self.api_url = api_url or os.getenv("ARGILLA_API_URL", "http://localhost:6900")
        self.api_key = api_key or os.getenv("ARGILLA_API_KEY", "admin.apikey")
        self.client = None
        self.datasets = {}
        self.user_id = "default_user"
        
    def connect(self) -> bool:
        """
        Connect to the Argilla server using provided credentials.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Argilla at {self.api_url}")
            self.client = rg.Argilla(api_url=self.api_url, api_key=self.api_key)
            
            # Test connection by getting current user
            user_info = self.client.me
            self.user_id = getattr(user_info, 'id', self.user_id)
            logger.info(f"Successfully connected to Argilla as user: {self.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Argilla: {str(e)}")
            return False
    
    def initialize_datasets(self) -> bool:
        """
        Initialize required datasets in Argilla according to the revised schema.
        Creates the datasets if they don't exist.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            if not self.client:
                success = self.connect()
                if not success:
                    return False
            
            # Dictionary to track dataset initialization status
            dataset_creation_results = {}
            
            # Initialize Records Dataset with integrated prompt fields
            dataset_creation_results["records"] = self._initialize_records_dataset()
            
            # Initialize Generator Prompts Dataset
            dataset_creation_results["generator_prompts"] = self._initialize_generator_prompts_dataset()
            
            # Initialize Evaluator Prompts Dataset
            dataset_creation_results["evaluator_prompts"] = self._initialize_evaluator_prompts_dataset()
            
            # Initialize Rounds Dataset
            dataset_creation_results["rounds"] = self._initialize_rounds_dataset()
            
            # Initialize Prompt Lineage Dataset
            dataset_creation_results["prompt_lineage"] = self._initialize_prompt_lineage_dataset()

            # Initialize Outputs Dataset
            dataset_creation_results["outputs"] = self._initialize_outputs_dataset()

            # Initialize Evaluations Dataset
            dataset_creation_results["evaluations"] = self._initialize_evaluations_dataset()
            
            # Check if all datasets were initialized successfully
            if not all(dataset_creation_results.values()):
                logger.error("Not all datasets were properly initialized")
                failed_datasets = [k for k, v in dataset_creation_results.items() if not v]
                logger.error(f"Failed datasets: {', '.join(failed_datasets)}")
                return False
            
            logger.info("All datasets initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing datasets: {str(e)}")
            return False
    
    def store_generated_content(self, input_data: str, content: str, 
                               prompt_id: str, round_num: int) -> Optional[str]:
        """
        Store generated content in the outputs dataset.
        
        Args:
            input_data: The input data used to generate content
            content: The generated content
            prompt_id: ID of the prompt used to generate the content
            round_num: Current round number
            
        Returns:
            str: ID of the stored record, or None if storage failed
        """
        try:
            if not self.client or "outputs" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Create a unique ID for this output
            output_id = str(uuid.uuid4())
            
            # Get the prompt text for reference, but continue even if not found
            prompt_text = self._get_prompt_text(prompt_id) or "Unknown prompt"
            
            # Create a record with properly structured data
            record = rg.Record(
                fields={
                    "input": input_data,
                    "generated_content": content,
                    "prompt_used": prompt_text
                },
                metadata={
                    "prompt_id": prompt_id,
                    "round": str(round_num),
                    "timestamp": datetime.now().isoformat(),
                    "output_id": output_id
                }
            )
            
            # Ensure the dataset reference is valid and log the record
            try:
                dataset = self.datasets["outputs"]
                dataset.records.log([record])
                logger.info(f"Stored generated content with ID: {output_id}")
                return output_id
            except Exception as e:
                logger.error(f"Error logging record to dataset: {str(e)}")
                
                # Try to reinitialize and retry if the initial attempt failed
                logger.info("Attempting to reconnect and retry...")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["outputs"].records.log([record])
                        logger.info(f"Successfully stored content after retry with ID: {output_id}")
                        return output_id
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error storing generated content: {str(e)}")
            return None
    
    def store_evaluation(self, output_id: str, score: int, feedback: str, 
                        improved_output: str, is_human: bool = False) -> bool:
        """
        Store evaluation of generated content.
        
        Args:
            output_id: ID of the output being evaluated
            score: Numerical score (1-5)
            feedback: Textual feedback
            improved_output: Improved version of the output
            is_human: Whether this is a human evaluation (default: False)
            
        Returns:
            bool: True if storage is successful, False otherwise
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return False
            
            # Check if evaluations dataset exists and is properly initialized
            if "evaluations" not in self.datasets or self.datasets["evaluations"] is None:
                logger.info("Evaluations dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return False
                
            # Confirm that we actually have the dataset now
            if self.datasets["evaluations"] is None:
                logger.error("Evaluations dataset still not available after initialization")
                return False
            
            # Get the original output
            output = self._get_output(output_id)
            if not output:
                logger.error(f"Output with ID {output_id} not found")
                return False
            
            # Safely extract fields and metadata
            input_data = ""
            generated_content = ""
            prompt_id = ""
            
            try:
                # Extract fields based on the structure
                if isinstance(output, dict):
                    # Try to get fields from the fields dictionary
                    if "fields" in output and isinstance(output["fields"], dict):
                        input_data = output["fields"].get("input", "")
                        generated_content = output["fields"].get("generated_content", "")
                    # Or try to get them directly from the output dict
                    else:
                        input_data = output.get("input", "")
                        generated_content = output.get("generated_content", "")
                    
                    # Try to get metadata
                    if "metadata" in output and isinstance(output["metadata"], dict):
                        prompt_id = output["metadata"].get("prompt_id", "unknown")
                    else:
                        prompt_id = output.get("prompt_id", "unknown")
                # Handle object-based structure
                elif hasattr(output, "fields"):
                    fields = getattr(output, "fields", {})
                    if isinstance(fields, dict):
                        input_data = fields.get("input", "")
                        generated_content = fields.get("generated_content", "")
                    else:
                        input_data = getattr(fields, "input", "")
                        generated_content = getattr(fields, "generated_content", "")
                    
                    metadata = getattr(output, "metadata", {})
                    if isinstance(metadata, dict):
                        prompt_id = metadata.get("prompt_id", "unknown")
                    else:
                        prompt_id = getattr(metadata, "prompt_id", "unknown")
                else:
                    logger.warning(f"Unexpected output structure: {type(output)}")
                    # Set default values if we can't extract the real ones
                    input_data = str(output)
                    generated_content = str(output)
                    prompt_id = "unknown"
            except Exception as e:
                logger.error(f"Error extracting data from output: {str(e)}")
                # Set default values
                input_data = ""
                generated_content = ""
                prompt_id = "unknown"
            
            # Create a unique ID for this evaluation
            evaluator_id = "human" if is_human else "ai_evaluator"
            
            # Ensure score is an integer in the valid range [1, 2, 3, 4, 5]
            # strip any non-numeric characters and convert to int
            score = ''.join(char for char in str(score) if char.isdigit())
            try:
                int_score = int(score)
                if int_score < 1:
                    int_score = 1
                    logger.warning(f"Score {score} was below minimum, setting to 1")
                elif int_score > 5:
                    int_score = 5
                    logger.warning(f"Score {score} was above maximum, setting to 5")
            except (ValueError, TypeError):
                logger.warning(f"Invalid score {score}, defaulting to 3")
                int_score = 3
            
            # Create responses with integer score
            responses = [
                rg.Response(question_name="score", value=int_score, user_id=self.user_id),
                rg.Response(question_name="feedback", value=feedback, user_id=self.user_id),
                rg.Response(question_name="improved_output", value=improved_output, user_id=self.user_id)
            ]
            
            # Create a record with standard metadata structure - remove the status field
            metadata = {
                "output_id": output_id,
                "prompt_id": prompt_id,
                "evaluator_id": evaluator_id,
                "is_human": "1" if is_human else "0",
                "timestamp": datetime.now().isoformat()
                # Removed the 'status' field as it's not defined in the dataset schema
            }
            
            record = rg.Record(
                fields={
                    "input": input_data,
                    "generated_content": generated_content,
                    "evaluation_content": ""  # This will be filled by responses
                },
                responses=responses,
                metadata=metadata
            )
            
            try:
                # Add the record to the dataset
                self.datasets["evaluations"].records.log([record])
                logger.info(f"Stored evaluation for output ID: {output_id}")
                return True
            except Exception as log_error:
                logger.error(f"Error logging evaluation record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry evaluation logging")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["evaluations"].records.log([record])
                        logger.info(f"Successfully stored evaluation after retry")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return False
            
        except Exception as e:
            logger.error(f"Error storing evaluation: {str(e)}")
            return False
    
    def store_human_feedback(self, output_id: str, score: int, feedback: str, 
                           improved_output: str) -> bool:
        """
        Store human feedback on AI evaluations (wrapper for store_evaluation).
        
        Args:
            output_id: ID of the output being evaluated
            score: Numerical score (1-5)
            feedback: Textual feedback
            improved_output: Improved version suggested by human
            
        Returns:
            bool: True if storage is successful, False otherwise
        """
        return self.store_evaluation(output_id, score, feedback, improved_output, is_human=True)
    
    def store_prompt(self, prompt_text: str, model: str, purpose: str, 
                   parent_prompt_id: Optional[str] = None, generation: int = 0) -> Optional[str]:
        """
        Store a prompt in the prompts dataset.
        
        Args:
            prompt_text: The prompt text
            model: The model the prompt is for
            purpose: The purpose of the prompt
            parent_prompt_id: ID of the parent prompt (if any)
            generation: Generation number
            
        Returns:
            str: ID of the stored prompt, or None if storage failed
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Use generator_prompts dataset instead of prompts
            dataset_name = "generator_prompts" if purpose == "generator" else "evaluator_prompts"
            
            # Check if dataset exists and is properly initialized
            if dataset_name not in self.datasets or self.datasets[dataset_name] is None:
                logger.info(f"{dataset_name} dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
                
            # Confirm that we actually have the dataset now
            if self.datasets[dataset_name] is None:
                logger.error(f"{dataset_name} dataset still not available after initialization")
                return None
            
            # Create a unique ID for this prompt
            prompt_id = str(uuid.uuid4())
            
            # Create a record
            record = rg.Record(
                fields={
                    "content": prompt_text
                },
                metadata={
                    "prompt_id": prompt_id,
                    "parent_prompt_id": parent_prompt_id or "root",
                    "avg_score_metadata": "0.0" if dataset_name == "generator_prompts" else "",
                    "rounds_survived": "0" if dataset_name == "generator_prompts" else "",
                    "active_status": "true",
                    "created_at": datetime.now().isoformat(),
                    "version": str(generation if generation > 0 else 1)
                }
            )
            
            try:
                # Add the record to the dataset
                self.datasets[dataset_name].records.log([record])
                logger.info(f"Stored {purpose} prompt with ID: {prompt_id}")
                return prompt_id
            except Exception as log_error:
                logger.error(f"Error logging prompt record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry prompt logging")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets[dataset_name].records.log([record])
                        logger.info(f"Successfully stored prompt after retry")
                        return prompt_id
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error storing prompt: {str(e)}")
            return None
    
    def update_prompt_performance(self, prompt_id: str, avg_score: float, survived: bool = False) -> bool:
        """
        Update the performance metrics for a prompt.
        
        Args:
            prompt_id: ID of the prompt
            avg_score: Average score across all outputs
            survived: Whether this prompt survived to the next generation
            
        Returns:
            bool: True if update is successful, False otherwise
        """
        try:
            if not self.client or "generator_prompts" not in self.datasets:  # Changed from "prompts" to "generator_prompts"
                success = self.initialize_datasets()
                if not success:
                    return False
            
            # Get the prompt record - changed from "prompts" to "generator_prompts"
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            records = self.datasets["generator_prompts"].records(query=query).to_list(flatten=True)
            
            if not records:
                logger.error(f"Prompt with ID {prompt_id} not found")
                return False
            
            # Create responses for update
            responses = [
                rg.Response(question_name="average_score", value=avg_score, user_id=self.user_id),
                rg.Response(question_name="survived", value=survived, user_id=self.user_id)
            ]
            
            # Update the record
            record = records[0]
            updated_record = rg.Record(
                id=record["id"],
                responses=responses
            )
            
            # Add the updated record to the dataset - changed from "prompts" to "generator_prompts"
            self.datasets["generator_prompts"].records.log([updated_record])
            
            logger.info(f"Updated performance for prompt ID: {prompt_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prompt performance: {str(e)}")
            return False
    
    def get_current_data_for_annotation(self, round_id: str, limit: int = 20) -> Optional[pd.DataFrame]:
        """
        Get the current data for human annotation based on round ID.
        
        Args:
            round_id: ID of the round to get data for
            limit: Maximum number of records to return
            
        Returns:
            pd.DataFrame: DataFrame containing the data for annotation
        """
        try:
            if not self.client:
                success = self.connect()
                if not success:
                    return None
                    
            # Ensure datasets are initialized
            if "records" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Query for records from the specified round
            round_filter = rg.Filter(("metadata.round_id", "==", round_id))
            query = rg.Query(filter=round_filter)
            records = self.datasets["records"].records(query=query).to_list(flatten=True)
            
            # Limit the number of records
            records = records[:limit]
            
            if not records:
                logger.warning(f"No records found for round {round_id}")
                return pd.DataFrame()
            
            # Build a DataFrame with the required columns
            rows = []
            for record in records:
                # Extract AI and human evaluations from the responses
                ai_score, ai_feedback, ai_improved_output = None, "", ""
                human_score, human_feedback, human_improved_output = None, "", ""
                
                if "responses" in record:
                    for response in record["responses"]:
                        question_name = response["question"]["name"]
                        if question_name == "ai_score":
                            ai_score = float(response["value"])
                        elif question_name == "ai_feedback":
                            ai_feedback = response["value"]
                        elif question_name == "ai_improved_output":
                            ai_improved_output = response["value"]
                        elif question_name == "human_score":
                            human_score = float(response["value"]) if response["value"] is not None else None
                        elif question_name == "human_feedback":
                            human_feedback = response["value"]
                        elif question_name == "human_improved_output":
                            human_improved_output = response["value"]
                
                row = {
                    "record_id": record["id"],
                    "input": record["fields"]["input"],
                    "content": record["fields"]["content"],
                    "ai_score": ai_score,
                    "ai_feedback": ai_feedback,
                    "ai_improved_output": ai_improved_output,
                    "human_score": human_score,
                    "human_feedback": human_feedback,
                    "human_improved_output": human_improved_output,
                    "generator_prompt_id": record["metadata"]["generator_prompt_id"],
                    "evaluator_prompt_id": record["metadata"]["evaluator_prompt_id"],
                    "is_validated": record["metadata"]["is_validated"] == "True"
                }
                rows.append(row)
            
            # Create the DataFrame
            df = pd.DataFrame(rows)
            
            logger.info(f"Retrieved {len(df)} records for annotation")
            return df
            
        except Exception as e:
            logger.error(f"Error getting data for annotation: {str(e)}")
            return None
    
    def get_performance_metrics(self, max_rounds: int = 10) -> Dict[str, Any]:
        """
        Get performance metrics for visualization.
        
        Args:
            max_rounds: Maximum number of rounds to include
            
        Returns:
            Dict[str, Any]: Dictionary containing performance metrics
        """
        try:
            if not self.client:
                success = self.connect()
                if not success:
                    return {"rounds": [], "prompts": [], "scores": [], "prompt_survivorship": {}, "moving_avg": []}
                    
            # Ensure datasets are initialized
            if not all(k in self.datasets for k in ["records", "generator_prompts", "rounds"]):
                success = self.initialize_datasets()
                if not success:
                    return {"rounds": [], "prompts": [], "scores": [], "prompt_survivorship": {}, "moving_avg": []}
            
            # Get all records with evaluations
            all_records = self.datasets["records"].records().to_list(flatten=True)
            
            # Get all generator prompts
            all_generator_prompts = self.datasets["generator_prompts"].records().to_list(flatten=True)
            
            # Get all rounds
            all_rounds = self.datasets["rounds"].records().to_list(flatten=True)
            
            # Calculate metrics
            metrics = {
                "rounds": [],
                "prompts": [],
                "scores": [],
                "prompt_survivorship": {}
            }
            
            # Process generator prompts
            for prompt in all_generator_prompts:
                prompt_id = prompt["metadata"].get("prompt_id", "unknown")
                parent_id = prompt["metadata"].get("parent_prompt_id", "root")
                avg_score = float(prompt["metadata"].get("average_score", 0))
                rounds_survived = int(prompt["metadata"].get("rounds_survived", 0))
                is_active = prompt["metadata"].get("is_active", "false").lower() == "true"
                
                # Track prompt metrics
                metrics["prompts"].append({
                    "id": prompt_id,
                    "parent_id": parent_id,
                    "avg_score": avg_score,
                    "rounds_survived": rounds_survived,
                    "is_active": is_active,
                    "content": prompt["fields"]["content"]
                })
                
                # Track survivorship
                if prompt_id not in metrics["prompt_survivorship"]:
                    metrics["prompt_survivorship"][prompt_id] = {
                        "generations": [rounds_survived],
                        "scores": [avg_score]
                    }
                else:
                    metrics["prompt_survivorship"][prompt_id]["generations"].append(rounds_survived)
                    metrics["prompt_survivorship"][prompt_id]["scores"].append(avg_score)
            
            # Process records and extract scores
            for record in all_records:
                round_id = record["metadata"].get("round_id", "unknown")
                generator_prompt_id = record["metadata"].get("generator_prompt_id", "unknown")
                
                # Extract AI score from responses
                ai_score = None
                if "responses" in record:
                    for response in record["responses"]:
                        question = response.get("question", {})
                        question_name = question.get("name", "")
                        if question_name == "score":
                            value = response.get("value")
                            ai_score = float(value) if value is not None else None
                        elif question_name == "feedback":
                            ai_feedback = response.get("value", "")
                        elif question_name == "improved_output":
                            ai_improved_output = response.get("value", "")
                
                if ai_score is not None:
                    # Find the round number for this record
                    round_number = None
                    for round_data in all_rounds:
                        if round_data["metadata"].get("round_id", "") == round_id:
                            round_number = int(round_data["fields"].get("number", 0))
                            break
                    
                    if round_number is not None:
                        metrics["rounds"].append(round_number)
                        metrics["scores"].append(ai_score)
            
            # Calculate moving averages
            if metrics["scores"]:
                window_size = min(5, len(metrics["scores"]))
                metrics["moving_avg"] = np.convolve(metrics["scores"], np.ones(window_size)/window_size, mode='valid').tolist()
            else:
                metrics["moving_avg"] = []
            
            logger.info(f"Retrieved performance metrics with {len(metrics['prompts'])} prompts")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {"rounds": [], "prompts": [], "scores": [], "prompt_survivorship": {}, "moving_avg": []}
    
    def get_prompt_history(self) -> Optional[pd.DataFrame]:
        """
        Get the history of prompts across generations.
        
        Returns:
            pd.DataFrame: DataFrame containing prompt history
        """
        try:
            if not self.client or "generator_prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Get all generator prompts
            all_prompts = self.datasets["generator_prompts"].records().to_list(flatten=True)
            
            rows = []
            for prompt in all_prompts:
                row = {
                    "prompt_id": prompt["metadata"].get("prompt_id", "unknown"),
                    "parent_prompt_id": prompt["metadata"].get("parent_prompt_id", "root"),
                    "content": prompt["fields"]["content"],
                    "average_score": float(prompt["metadata"].get("average_score", 0)),
                    "rounds_survived": int(prompt["metadata"].get("rounds_survived", 0)),
                    "is_active": prompt["metadata"].get("is_active", "false").lower() == "true",
                    "version": int(prompt["metadata"].get("version", 1)),
                    "created_at": prompt["metadata"].get("created_at", datetime.now().isoformat())
                }
                rows.append(row)
            
            # Create the DataFrame
            df = pd.DataFrame(rows)
            
            # Sort by version and created_at
            if not df.empty:
                df = df.sort_values(by=["version", "created_at"])
            
            logger.info(f"Retrieved prompt history with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error getting prompt history: {str(e)}")
            return None
    
    def get_current_best_prompts(self, top_n: int = 4) -> List[str]:
        """
        Get the current best performing prompts.
        
        Args:
            top_n: Number of top prompts to return
            
        Returns:
            List[str]: List of prompt IDs
        """
        try:
            if not self.client or "generator_prompts" not in self.datasets:  # Changed from "prompts" to "generator_prompts"
                success = self.initialize_datasets()
                if not success:
                    return []
            
            # Get all prompts - changed from "prompts" to "generator_prompts"
            all_prompts = self.datasets["generator_prompts"].records().to_list(flatten=True)
            
            # Extract prompt IDs and scores
            prompt_scores = []
            for prompt in all_prompts:
                prompt_id = prompt["metadata"]["prompt_id"]
                
                # Extract avg_score from responses or metadata
                avg_score = 0
                if "responses" in prompt:
                    for response in prompt["responses"]:
                        question = response.get("question", {})
                        question_name = question.get("name", "")
                        if question_name == "average_score":
                            avg_score = float(response["value"])
                            break
                # If not in responses, try metadata
                if avg_score == 0 and "metadata" in prompt:
                    avg_score = float(prompt["metadata"].get("avg_score_metadata", 0))
                
                prompt_scores.append((prompt_id, avg_score))
            
            # Sort by score (descending) and take top N
            prompt_scores.sort(key=lambda x: x[1], reverse=True)
            top_prompts = [ps[0] for ps in prompt_scores[:top_n]]
            
            logger.info(f"Retrieved top {len(top_prompts)} prompts")
            return top_prompts
            
        except Exception as e:
            logger.error(f"Error getting best prompts: {str(e)}")
            return []
    
    def _get_output(self, output_id: str) -> Optional[Dict]:
        """
        Helper method to get an output by ID.
        
        Args:
            output_id: ID of the output
            
        Returns:
            Dict: Output record, or None if not found
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if outputs dataset exists and is properly initialized
            if "outputs" not in self.datasets or self.datasets["outputs"] is None:
                logger.info("Outputs dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
                
            # Confirm that we actually have the dataset now
            if self.datasets["outputs"] is None:
                logger.error("Outputs dataset still not available after initialization")
                return None
                
            # Query for the output using Filter
            output_filter = rg.Filter(("metadata.output_id", "==", output_id))
            query = rg.Query(filter=output_filter)
            
            try:
                records_iterator = self.datasets["outputs"].records(query=query)
                records_list = records_iterator.to_list(flatten=True)
                
                if not records_list:
                    logger.warning(f"No output found with ID: {output_id}")
                    return None
                    
                # Return the first matching record
                return records_list[0]
            except Exception as query_error:
                logger.error(f"Error querying for output: {str(query_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry output query")
                if self.connect() and self.initialize_datasets():
                    try:
                        records_iterator = self.datasets["outputs"].records(query=query)
                        records_list = records_iterator.to_list(flatten=True)
                        return records_list[0] if records_list else None
                    except Exception as retry_error:
                        logger.error(f"Retry query failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error getting output: {str(e)}")
            return None
    
    def _get_prompt_text(self, prompt_id: str) -> Optional[str]:
        """
        Helper method to get a prompt text by ID.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            str: Prompt text, or None if not found
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None

            # Check if generator_prompts dataset exists and is properly initialized
            if "generator_prompts" not in self.datasets or self.datasets["generator_prompts"] is None:
                logger.info("Generator prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
                
            # Confirm that we actually have the dataset now
            if self.datasets["generator_prompts"] is None:
                logger.error("Generator prompts dataset still not available after initialization")
                return None

            # Query for the prompt using the Filter API - changed from "prompts" to "generator_prompts"
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            
            try:
                records_iterator = self.datasets["generator_prompts"].records(query=query)
                records_list = records_iterator.to_list(flatten=True)
                
                if not records_list:
                    logger.warning(f"No prompt found with ID: {prompt_id}")
                    return None
                    
                # Return the prompt text - handle possible data structure differences
                record = records_list[0]
                if isinstance(record, dict) and "fields" in record:
                    # Changed from "prompt_text" to "content" to match the field name in generator_prompts
                    return record["fields"].get("content", None)
                elif hasattr(record, 'fields'):
                    # Changed from "prompt_text" to "content" to match the field name in generator_prompts
                    return getattr(record.fields, 'content', None)
                else:
                    logger.warning(f"Prompt found but could not extract text, record structure: {record}")
                    return None
            except Exception as query_error:
                logger.error(f"Error querying for prompt: {str(query_error)}")
                
                # Try once more with a fresh connection
                logger.info("Attempting to reconnect and retry prompt query")
                if self.connect() and self.initialize_datasets():
                    try:
                        records_iterator = self.datasets["generator_prompts"].records(query=query)
                        records_list = records_iterator.to_list(flatten=True)
                        if records_list:
                            record = records_list[0]
                            if isinstance(record, dict) and "fields" in record:
                                # Changed from "prompt_text" to "content" to match the field name
                                return record["fields"].get("content", None)
                            elif hasattr(record, 'fields'):
                                # Changed from "prompt_text" to "content" to match the field name
                                return getattr(record.fields, 'content', None)
                    except Exception as retry_error:
                        logger.error(f"Retry query failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error getting prompt text: {str(e)}")
            return None
    
    def _get_latest_evaluation(self, output_id: str) -> Optional[Dict]:
        """
        Helper method to get the latest evaluation for an output.
        
        Args:
            output_id: ID of the output
            
        Returns:
            Dict: Evaluation record, or None if not found
        """
        try:
            if "evaluations" not in self.datasets:
                logger.info("Evaluations dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    return None
                    
            # Query for evaluations of this output
            output_filter = rg.Filter(("metadata.output_id", "==", output_id))
            query = rg.Query(filter=output_filter)
            
            try:
                records_iterator = self.datasets["evaluations"].records(query=query)
                records_list = records_iterator.to_list(flatten=True)
                
                if not records_list:
                    logger.debug(f"No evaluations found for output ID: {output_id}")
                    return None
                    
                # Sort by timestamp (descending) and return the first (latest)
                records_list.sort(key=lambda x: x["metadata"]["timestamp"] if isinstance(x, dict) and "metadata" in x else 
                                   (x.metadata.timestamp if hasattr(x, 'metadata') and hasattr(x.metadata, 'timestamp') else ""), 
                                   reverse=True)
                
                latest = records_list[0]
                
                # Extract the relevant fields from responses
                evaluation = {
                    "score": 0,
                    "feedback": "",
                    "improved_output": ""
                }
                
                # Handle both dict and object access patterns for records
                if isinstance(latest, dict) and "responses" in latest:
                    for response in latest["responses"]:
                        question = response.get("question", {})
                        question_name = question.get("name", "")
                        if question_name == "score":
                            value = response.get("value")
                            evaluation["score"] = float(value) if value is not None else None
                        elif question_name == "feedback":
                            evaluation["feedback"] = response.get("value", "")
                        elif question_name == "improved_output":
                            evaluation["improved_output"] = response.get("value", "")
                elif hasattr(latest, 'responses'):
                    for response in latest.responses:
                        question = response.get("question", {})
                        question_name = question.get("name", "")
                        if hasattr(response, 'question') and hasattr(response.question, 'name'):
                            if response.question.name == "score":
                                value = response.get("value")
                                evaluation["score"] = float(value) if value is not None else None
                            elif response.question.name == "feedback":
                                evaluation["feedback"] = response.get("value", "")
                            elif response.question.name == "improved_output":
                                evaluation["improved_output"] = response.get("value", "")
                
                return evaluation
                
            except Exception as query_error:
                logger.error(f"Error querying for evaluations: {str(query_error)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting latest evaluation: {str(e)}")
            return None
    
    def get_validated_evaluations(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Get evaluations that have been validated by humans.
        
        Args:
            limit: Maximum number of evaluations to return (default: 10)
            
        Returns:
            pd.DataFrame: DataFrame containing the validated evaluations, or None if error
        """
        try:
            # Ensure connection to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if evaluations dataset exists and is properly initialized
            if "evaluations" not in self.datasets or self.datasets["evaluations"] is None:
                logger.info("Evaluations dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Query for evaluations with human validation
            # Filter evaluations where is_human = "1"
            human_filter = rg.Filter(("metadata.is_human", "==", "1"))
            query = rg.Query(filter=human_filter)
            
            try:
                records = self.datasets["evaluations"].records(query=query).to_list(flatten=True)
                
                if not records:
                    logger.info("No validated evaluations found")
                    return pd.DataFrame()
                
                # Sort by timestamp (most recent first)
                records.sort(key=lambda x: x.get("metadata", {}).get("timestamp", ""), reverse=True)
                
                # Limit the number of records
                records = records[:limit]
                
                # Build DataFrame with required columns
                rows = []
                for record in records:
                    # Safely extract metadata
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                        
                    # Extract relevant data from the record
                    output_id = metadata.get("output_id", "")
                    prompt_id = metadata.get("prompt_id", "")
                    timestamp = metadata.get("timestamp", "")
                    
                    # Extract score, feedback and improved output from responses
                    score, feedback, improved_output = None, "", ""
                    responses = record.get("responses", [])
                    if responses:
                        for response in responses:
                            question = response.get("question", {})
                            question_name = question.get("name", "")
                            if question_name == "score":
                                value = response.get("value")
                                score = float(value) if value is not None else None
                            elif question_name == "feedback":
                                feedback = response.get("value", "")
                            elif question_name == "improved_output":
                                improved_output = response.get("value", "")
                    
                    # Get the original input and generated content
                    fields = record.get("fields", {})
                    if not isinstance(fields, dict):
                        fields = {}
                        
                    input_data = fields.get("input", "")
                    generated_content = fields.get("generated_content", "")
                    
                    row = {
                        "output_id": output_id,
                        "prompt_id": prompt_id,
                        "input": input_data,
                        "generated_content": generated_content,
                        "score": score,
                        "feedback": feedback,
                        "improved_output": improved_output,
                        "timestamp": timestamp
                    }
                    rows.append(row)
                
                # Create DataFrame
                df = pd.DataFrame(rows)
                
                logger.info(f"Retrieved {len(df)} validated evaluations")
                return df
                
            except Exception as query_error:
                logger.error(f"Error querying for validated evaluations: {str(query_error)}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting validated evaluations: {str(e)}")
            return None
    
    def _initialize_records_dataset(self) -> bool:
        """
        Initialize the Records dataset with integrated prompt fields.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Check if dataset exists
            try:
                # Fix: Use get_dataset instead of datasets()
                records_dataset = self.client.get_dataset("archer_records")
                if records_dataset is not None:
                    self.datasets["records"] = records_dataset
                    logger.info("Found existing records dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new records dataset: {str(e)}")
                records_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="input", title="Input Data"),
                        rg.TextField(name="content", title="Generated Content"),
                        rg.TextField(name="generator_prompt", title="Generator Prompt"),  # New field
                        rg.TextField(name="evaluator_prompt", title="Evaluator Prompt")   # New field
                    ],
                    questions=[
                        rg.RatingQuestion(
                            name="ai_score",
                            values=[1, 2, 3, 4, 5],
                            title="AI Quality Score",
                            description="AI-generated quality score",
                            required=True
                        ),
                        rg.TextQuestion(
                            name="ai_feedback",
                            title="AI Feedback",
                            description="AI-generated feedback",
                            required=True,
                            use_markdown=True
                        ),
                        rg.TextQuestion(
                            name="ai_improved_output",
                            title="AI Improved Output",
                            description="AI-suggested improved version",
                            required=True,
                            use_markdown=True
                        ),
                        rg.RatingQuestion(
                            name="human_score",
                            values=[1, 2, 3, 4, 5],
                            title="Human Quality Score",
                            description="Human-generated quality score",
                            required=False
                        ),
                        rg.TextQuestion(
                            name="human_feedback",
                            title="Human Feedback",
                            description="Human-generated feedback",
                            required=False,
                            use_markdown=True
                        ),
                        rg.TextQuestion(
                            name="human_improved_output",
                            title="Human Improved Output",
                            description="Human-suggested improved version",
                            required=False,
                            use_markdown=True
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="prompt_generation", title="Prompt Generation"),  # New field
                        rg.TermsMetadataProperty(name="round_id", title="Round ID"),
                        rg.TermsMetadataProperty(name="timestamp", title="Timestamp"),
                        rg.TermsMetadataProperty(name="validated_status", title="Is Validated")
                    ]
                )
                
                logger.info("Creating archer_records dataset...")
                new_dataset = rg.Dataset(name="archer_records", settings=records_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again with the correct method
                self.datasets["records"] = self.client.get_dataset("archer_records")
                if self.datasets["records"] is None:
                    raise Exception("Failed to create archer_records dataset")
                
                logger.info("Created new records dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing records dataset: {str(e)}")
            return False
    
    def _initialize_generator_prompts_dataset(self) -> bool:
        """
        Initialize the Generator Prompts dataset.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Check if dataset exists
            try:
                # Fix: Use get_dataset instead of datasets()
                prompts_dataset = self.client.get_dataset("archer_generator_prompts")
                if prompts_dataset is not None:
                    self.datasets["generator_prompts"] = prompts_dataset
                    logger.info("Found existing generator prompts dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new generator prompts dataset: {str(e)}")
                prompts_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="content", title="Prompt Content")
                    ],
                    questions=[
                        # Add at least one question to satisfy Argilla requirements
                        rg.RatingQuestion(
                            name="average_score", 
                            values=[1, 2, 3, 4, 5],
                            title="Average Score",
                            description="Average performance score of this prompt",
                            required=True
                        ),
                        rg.LabelQuestion(
                            name="survived",
                            labels={"TRUE": "Yes", "FALSE": "No"},
                            title="Survived",
                            description="Whether this prompt survived to the next round",
                            required=False
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                        rg.TermsMetadataProperty(name="parent_prompt_id", title="Parent Prompt ID"),
                        rg.TermsMetadataProperty(name="avg_score_metadata", title="Average Score"),
                        rg.TermsMetadataProperty(name="rounds_survived", title="Rounds Survived"),
                        rg.TermsMetadataProperty(name="active_status", title="Is Active"),  # Renamed from "is_active" to avoid potential conflict
                        rg.TermsMetadataProperty(name="created_at", title="Created At"),
                        rg.TermsMetadataProperty(name="version", title="Version")
                    ]
                )
                
                logger.info("Creating archer_generator_prompts dataset...")
                new_dataset = rg.Dataset(name="archer_generator_prompts", settings=prompts_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again with the correct method
                self.datasets["generator_prompts"] = self.client.get_dataset("archer_generator_prompts")
                if self.datasets["generator_prompts"] is None:
                    raise Exception("Failed to create archer_generator_prompts dataset")
                
                logger.info("Created new generator prompts dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing generator prompts dataset: {str(e)}")
            return False
    
    def _initialize_evaluator_prompts_dataset(self) -> bool:
        """
        Initialize the Evaluator Prompts dataset.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Check if dataset exists
            try:
                # Fix: Use get_dataset instead of datasets()
                prompts_dataset = self.client.get_dataset("archer_evaluator_prompts")
                if prompts_dataset is not None:
                    self.datasets["evaluator_prompts"] = prompts_dataset
                    logger.info("Found existing evaluator prompts dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new evaluator prompts dataset: {str(e)}")
                prompts_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="content", title="Prompt Content")
                    ],
                    questions=[
                        # Add questions to satisfy Argilla requirements
                        rg.LabelQuestion(
                            name="is_active",
                            labels={"TRUE": "Yes", "FALSE": "No"},
                            title="Is Active",
                            description="Whether this evaluator prompt is currently active",
                            required=True
                        ),
                        rg.TextQuestion(
                            name="notes",
                            title="Notes",
                            description="Additional notes about this evaluator prompt",
                            required=False,
                            use_markdown=True
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                        rg.TermsMetadataProperty(name="active_status", title="Is Active"),  # Renamed from "is_active" to avoid conflict
                        rg.TermsMetadataProperty(name="created_at", title="Created At"),
                        rg.TermsMetadataProperty(name="version", title="Version")
                    ]
                )
                
                logger.info("Creating archer_evaluator_prompts dataset...")
                new_dataset = rg.Dataset(name="archer_evaluator_prompts", settings=prompts_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again with the correct method
                self.datasets["evaluator_prompts"] = self.client.get_dataset("archer_evaluator_prompts")
                if self.datasets["evaluator_prompts"] is None:
                    raise Exception("Failed to create archer_evaluator_prompts dataset")
                
                logger.info("Created new evaluator prompts dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing evaluator prompts dataset: {str(e)}")
            return False
    
    def _initialize_rounds_dataset(self) -> bool:
        """
        Initialize the Rounds dataset.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Check if dataset exists
            try:
                # Fix: Use get_dataset instead of datasets()
                rounds_dataset = self.client.get_dataset("archer_rounds")
                if rounds_dataset is not None:
                    self.datasets["rounds"] = rounds_dataset
                    logger.info("Found existing rounds dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new rounds dataset: {str(e)}")
                rounds_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="number", title="Round Number"),
                        rg.TextField(name="status", title="Status")
                    ],
                    questions=[
                        # Add questions to satisfy Argilla requirements
                        rg.TextQuestion(
                            name="metrics_summary",
                            title="Metrics Summary",
                            description="Summary of round performance metrics",
                            required=False,
                            use_markdown=True
                        ),
                        rg.RatingQuestion(
                            name="overall_performance",
                            values=[1, 2, 3, 4, 5],
                            title="Overall Performance",
                            description="Overall performance rating of this round",
                            required=True
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="round_id", title="Round ID"),
                        rg.TermsMetadataProperty(name="start_time", title="Start Time"),
                        rg.TermsMetadataProperty(name="end_time", title="End Time"),
                        rg.TermsMetadataProperty(name="metrics", title="Metrics")
                    ]
                )
                
                logger.info("Creating archer_rounds dataset...")
                new_dataset = rg.Dataset(name="archer_rounds", settings=rounds_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again with the correct method
                self.datasets["rounds"] = self.client.get_dataset("archer_rounds")
                if self.datasets["rounds"] is None:
                    raise Exception("Failed to create archer_rounds dataset")
                
                logger.info("Created new rounds dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing rounds dataset: {str(e)}")
            return False
    
    def _initialize_prompt_lineage_dataset(self) -> bool:
        """
        Initialize the Prompt Lineage dataset.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Check if dataset exists
            try:
                # Fix: Use get_dataset instead of datasets()
                lineage_dataset = self.client.get_dataset("archer_prompt_lineage")
                if lineage_dataset is not None:
                    self.datasets["prompt_lineage"] = lineage_dataset
                    logger.info("Found existing prompt lineage dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new prompt lineage dataset: {str(e)}")
                lineage_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="change_reason", title="Change Reason")
                    ],
                    questions=[
                        # Add questions to satisfy Argilla requirements
                        rg.TextQuestion(
                            name="effectiveness",
                            title="Change Effectiveness",
                            description="Assessment of how effective the prompt change was",
                            required=False,
                            use_markdown=True
                        ),
                        rg.RatingQuestion(
                            name="improvement_rating",
                            values=[1, 2, 3, 4, 5],
                            title="Improvement Rating",
                            description="Rating of the improvement from parent to child prompt",
                            required=True
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="lineage_id", title="Lineage ID"),
                        rg.TermsMetadataProperty(name="parent_prompt_id", title="Parent Prompt ID"),
                        rg.TermsMetadataProperty(name="child_prompt_id", title="Child Prompt ID"),
                        rg.TermsMetadataProperty(name="round_id", title="Round ID"),
                        rg.TermsMetadataProperty(name="timestamp", title="Timestamp")
                    ]
                )
                
                logger.info("Creating archer_prompt_lineage dataset...")
                new_dataset = rg.Dataset(name="archer_prompt_lineage", settings=lineage_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again with the correct method
                self.datasets["prompt_lineage"] = self.client.get_dataset("archer_prompt_lineage")
                if self.datasets["prompt_lineage"] is None:
                    raise Exception("Failed to create archer_prompt_lineage dataset")
                
                logger.info("Created new prompt lineage dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing prompt lineage dataset: {str(e)}")
            return False
    
    def _initialize_outputs_dataset(self) -> bool:
        """
        Initialize dataset for storing model outputs.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if dataset already exists
            try:
                # Fix: Use get_dataset instead of datasets() which doesn't have list_datasets
                outputs_dataset = self.client.get_dataset("archer_outputs")
                if outputs_dataset is not None:
                    self.datasets["outputs"] = outputs_dataset
                    logger.info("Found existing outputs dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new outputs dataset: {str(e)}")
                
                # Define settings for the dataset
                outputs_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="input", title="Input Data"),
                        rg.TextField(name="generated_content", title="Generated Content"),
                        rg.TextField(name="prompt_used", title="Prompt Used")
                    ],
                    questions=[
                        rg.RatingQuestion(
                            name="quality_score",
                            values=[1, 2, 3, 4, 5],
                            title="Quality Score",
                            description="Rate the quality of the generated output",
                            required=True
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                        rg.TermsMetadataProperty(name="round", title="Round Number"),
                        rg.TermsMetadataProperty(name="output_id", title="Output ID"),
                        rg.TermsMetadataProperty(name="timestamp", title="Timestamp")
                    ]
                )
                
                logger.info("Creating archer_outputs dataset...")
                new_dataset = rg.Dataset(name="archer_outputs", settings=outputs_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again using the corrected method
                self.datasets["outputs"] = self.client.get_dataset("archer_outputs")
                if self.datasets["outputs"] is None:
                    raise Exception("Failed to create archer_outputs dataset")
                
                logger.info("Created new outputs dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing outputs dataset: {str(e)}")
            return False
    
    def _initialize_evaluations_dataset(self) -> bool:
        """
        Initialize the Evaluations dataset.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            # Check if dataset exists
            try:
                # Fix: Use get_dataset instead of datasets()
                evaluations_dataset = self.client.get_dataset("archer_evaluations")
                if evaluations_dataset is not None:
                    self.datasets["evaluations"] = evaluations_dataset
                    logger.info("Found existing evaluations dataset")
                    return True
                else:
                    raise Exception("Dataset returned is None")
            except Exception as e:
                logger.info(f"Creating new evaluations dataset: {str(e)}")
                evaluations_settings = rg.Settings(
                    fields=[
                        rg.TextField(name="input", title="Input Data"),
                        rg.TextField(name="generated_content", title="Generated Content"),
                        rg.TextField(name="evaluation_content", title="Evaluation Content")
                    ],
                    questions=[
                        rg.RatingQuestion(
                            name="score",
                            values=[1, 2, 3, 4, 5],
                            title="Quality Score",
                            description="Quality score for the generated content",
                            required=True
                        ),
                        rg.TextQuestion(
                            name="feedback",
                            title="Feedback",
                            description="Feedback on the generated content",
                            required=True,
                            use_markdown=True
                        ),
                        rg.TextQuestion(
                            name="improved_output",
                            title="Improved Output",
                            description="Improved version of the generated content",
                            required=True,
                            use_markdown=True
                        )
                    ],
                    metadata=[
                        rg.TermsMetadataProperty(name="output_id", title="Output ID"),
                        rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                        rg.TermsMetadataProperty(name="evaluator_id", title="Evaluator ID"),
                        rg.TermsMetadataProperty(name="is_human", title="Is Human"),
                        rg.TermsMetadataProperty(name="timestamp", title="Timestamp")
                    ]
                )
                
                logger.info("Creating archer_evaluations dataset...")
                new_dataset = rg.Dataset(name="archer_evaluations", settings=evaluations_settings)
                new_dataset.create()
                
                # Verify creation worked by fetching again using correct method
                self.datasets["evaluations"] = self.client.get_dataset("archer_evaluations")
                if self.datasets["evaluations"] is None:
                    raise Exception("Failed to create archer_evaluations dataset")
                
                logger.info("Created new evaluations dataset")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing evaluations dataset: {str(e)}")
            return False
    
    def store_record(self, input_data: str, content: str, 
                   generator_prompt: str, evaluator_prompt: str, 
                   prompt_generation: int, round_id: str) -> Optional[str]:
        """
        Store a record with integrated prompt information.
        
        Args:
            input_data: The input data
            content: The generated content
            generator_prompt: The actual generator prompt text
            evaluator_prompt: The actual evaluator prompt text
            prompt_generation: Generation number of the prompt
            round_id: Current round ID
            
        Returns:
            str: ID of the stored record, or None if storage failed
        """
        try:
            if not self.client or "records" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Create a unique ID for this record
            record_id = str(uuid.uuid4())
            
            # Create a record with the updated structure
            record = rg.Record(
                fields={
                    "input": input_data,
                    "content": content,
                    "generator_prompt": generator_prompt,
                    "evaluator_prompt": evaluator_prompt
                },
                metadata={
                    "prompt_generation": prompt_generation,
                    "round_id": round_id,
                    "timestamp": datetime.now().isoformat(),
                    "validated_status": "False"
                }
            )
            
            # Ensure the dataset reference is valid and log the record
            try:
                dataset = self.datasets["records"]
                dataset.records.log([record])
                logger.info(f"Stored record with ID: {record_id}")
                return record_id
            except Exception as e:
                logger.error(f"Error logging record to dataset: {str(e)}")
                
                # Try to reinitialize and retry if the initial attempt failed
                logger.info("Attempting to reconnect and retry...")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["records"].records.log([record])
                        logger.info(f"Successfully stored record after retry with ID: {record_id}")
                        return record_id
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error storing record: {str(e)}")
            return None
    
    def update_record_evaluation(self, record_id: str, ai_score: float, 
                               ai_feedback: str, ai_improved_output: str) -> bool:
        """
        Update a record with AI evaluation data.
        
        Args:
            record_id: ID of the record to update
            ai_score: AI-generated quality score (1-5)
            ai_feedback: AI-generated feedback
            ai_improved_output: AI-suggested improved version
            
        Returns:
            bool: True if update is successful, False otherwise
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return False
            
            # Check if records dataset exists and is properly initialized
            if "records" not in self.datasets or self.datasets["records"] is None:
                logger.info("Records dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return False
            
            # Get the existing record
            record = self._get_record(record_id)
            if not record:
                logger.error(f"Record with ID {record_id} not found")
                return False
            
            # Create responses
            responses = [
                rg.Response(question_name="ai_score", value=ai_score, user_id=self.user_id),
                rg.Response(question_name="ai_feedback", value=ai_feedback, user_id=self.user_id),
                rg.Response(question_name="ai_improved_output", value=ai_improved_output, user_id=self.user_id)
            ]
            
            # Update the record
            updated_record = rg.Record(
                id=record.get("id"),
                responses=responses
            )
            
            # Add the updated record to the dataset
            try:
                self.datasets["records"].records.log([updated_record])
                logger.info(f"Updated record with AI evaluation: {record_id}")
                return True
            except Exception as log_error:
                logger.error(f"Error logging updated record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["records"].records.log([updated_record])
                        logger.info(f"Successfully updated record after retry")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return False
            
        except Exception as e:
            logger.error(f"Error updating record evaluation: {str(e)}")
            return False
    
    def update_record_human_feedback(self, record_id: str, human_score: float, 
                                   human_feedback: str, human_improved_output: str) -> bool:
        """
        Update a record with human feedback data.
        
        Args:
            record_id: ID of the record to update
            human_score: Human-assigned quality score (1-5)
            human_feedback: Human feedback
            human_improved_output: Human-suggested improved version
            
        Returns:
            bool: True if update is successful, False otherwise
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return False
            
            # Check if records dataset exists and is properly initialized
            if "records" not in self.datasets or self.datasets["records"] is None:
                logger.info("Records dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return False
            
            # Get the existing record
            record = self._get_record(record_id)
            if not record:
                logger.error(f"Record with ID {record_id} not found")
                return False
            
            # Create responses
            responses = [
                rg.Response(question_name="human_score", value=human_score, user_id=self.user_id),
                rg.Response(question_name="human_feedback", value=human_feedback, user_id=self.user_id),
                rg.Response(question_name="human_improved_output", value=human_improved_output, user_id=self.user_id)
            ]
            
            # Update the record
            updated_record = rg.Record(
                id=record.get("id"),
                responses=responses,
                metadata={
                    "validated_status": "True"  # Updated to match renamed property
                }
            )
            
            # Add the updated record to the dataset
            try:
                self.datasets["records"].records.log([updated_record])
                logger.info(f"Updated record with human feedback: {record_id}")
                return True
            except Exception as log_error:
                logger.error(f"Error logging updated record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["records"].records.log([updated_record])
                        logger.info(f"Successfully updated record after retry")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return False
            
        except Exception as e:
            logger.error(f"Error updating record with human feedback: {str(e)}")
            return False
    
    def store_generator_prompt(self, content: str, parent_prompt_id: Optional[str] = None, 
                             version: int = 1) -> Optional[str]:
        """
        Store a generator prompt in the generator_prompts dataset.
        
        Args:
            content: The prompt content
            parent_prompt_id: ID of the parent prompt (if any)
            version: Version number of the prompt
            
        Returns:
            str: ID of the stored prompt, or None if storage failed
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if generator_prompts dataset exists and is properly initialized
            if "generator_prompts" not in self.datasets or self.datasets["generator_prompts"] is None:
                logger.info("Generator prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Create a unique ID for this prompt
            prompt_id = str(uuid.uuid4())
            
            # Create a record
            record = rg.Record(
                fields={
                    "content": content
                },
                metadata={
                    "prompt_id": prompt_id,
                    "parent_prompt_id": parent_prompt_id or "root",
                    "avg_score_metadata": "0.0",
                    "rounds_survived": "0",
                    "active_status": "true",  # Updated to match renamed property
                    "created_at": datetime.now().isoformat(),
                    "version": str(version)
                }
            )
            
            # Add the record to the dataset
            try:
                self.datasets["generator_prompts"].records.log([record])
                logger.info(f"Stored generator prompt with ID: {prompt_id}")
                return prompt_id
            except Exception as log_error:
                logger.error(f"Error logging generator prompt record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry generator prompt logging")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["generator_prompts"].records.log([record])
                        logger.info(f"Successfully stored generator prompt after retry")
                        return prompt_id
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error storing generator prompt: {str(e)}")
            return None
    
    def store_evaluator_prompt(self, content: str, version: int = 1) -> Optional[str]:
        """
        Store an evaluator prompt in the evaluator_prompts dataset.
        
        Args:
            content: The prompt content
            version: Version number of the prompt
            
        Returns:
            str: ID of the stored prompt, or None if storage failed
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if evaluator_prompts dataset exists and is properly initialized
            if "evaluator_prompts" not in self.datasets or self.datasets["evaluator_prompts"] is None:
                logger.info("Evaluator prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Create a unique ID for this prompt
            prompt_id = str(uuid.uuid4())
            
            # Create a record
            record = rg.Record(
                fields={
                    "content": content
                },
                metadata={
                    "prompt_id": prompt_id,
                    "active_status": "true",  # Updated to match renamed property
                    "created_at": datetime.now().isoformat(),
                    "version": str(version)
                }
            )
            
            # Add the record to the dataset
            try:
                self.datasets["evaluator_prompts"].records.log([record])
                logger.info(f"Stored evaluator prompt with ID: {prompt_id}")
                return prompt_id
            except Exception as log_error:
                logger.error(f"Error logging evaluator prompt record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry evaluator prompt logging")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["evaluator_prompts"].records.log([record])
                        logger.info(f"Successfully stored evaluator prompt after retry")
                        return prompt_id
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error storing evaluator prompt: {str(e)}")
            return None
    
    def update_generator_prompt_performance(self, prompt_id: str, avg_score: float, 
                                          rounds_survived: int, is_active: bool) -> bool:
        """
        Update the performance metrics for a generator prompt.
        
        Args:
            prompt_id: ID of the prompt
            avg_score: Average score across all outputs
            rounds_survived: Number of rounds the prompt has survived
            is_active: Whether the prompt is active
            
        Returns:
            bool: True if update is successful, False otherwise
        """
        try:
            if not self.client or "generator_prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return False
            
            # Get the prompt record
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            records = self.datasets["generator_prompts"].records(query=query).to_list(flatten=True)
            
            if not records:
                logger.error(f"Generator prompt with ID {prompt_id} not found")
                return False
            
            # Update the record
            record = records[0]
            updated_record = rg.Record(
                id=record["id"],
                metadata={
                    "avg_score_metadata": str(avg_score),
                    "rounds_survived": str(rounds_survived),
                    "active_status": str(is_active).lower()  # Updated to match renamed property
                }
            )
            
            # Add the updated record to the dataset
            try:
                self.datasets["generator_prompts"].records.log([updated_record])
                logger.info(f"Updated performance for generator prompt ID: {prompt_id}")
                return True
            except Exception as log_error:
                logger.error(f"Error updating generator prompt: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["generator_prompts"].records.log([updated_record])
                        logger.info(f"Successfully updated generator prompt after retry")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Retry update failed: {str(retry_error)}")
                
                return False
            
        except Exception as e:
            logger.error(f"Error updating generator prompt performance: {str(e)}")
            return False
    
    def update_evaluator_prompt_status(self, prompt_id: str, is_active: bool) -> bool:
        """
        Update the status of an evaluator prompt.
        
        Args:
            prompt_id: ID of the prompt
            is_active: Whether the prompt is active
            
        Returns:
            bool: True if update is successful, False otherwise
        """
        try:
            if not self.client or "evaluator_prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return False
            
            # Get the prompt record
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            records = self.datasets["evaluator_prompts"].records(query=query).to_list(flatten=True)
            
            if not records:
                logger.error(f"Evaluator prompt with ID {prompt_id} not found")
                return False
            
            # Update the record
            record = records[0]
            updated_record = rg.Record(
                id=record["id"],
                metadata={
                    "active_status": str(is_active).lower()  # Updated to match renamed property
                }
            )
            
            # Add the updated record to the dataset
            try:
                self.datasets["evaluator_prompts"].records.log([updated_record])
                logger.info(f"Updated status for evaluator prompt ID: {prompt_id}")
                return True
            except Exception as log_error:
                logger.error(f"Error updating evaluator prompt: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["evaluator_prompts"].records.log([updated_record])
                        logger.info(f"Successfully updated evaluator prompt after retry")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Retry update failed: {str(retry_error)}")
                
                return False
            
        except Exception as e:
            logger.error(f"Error updating evaluator prompt status: {str(e)}")
            return False
    
    def create_round(self, round_number: int) -> Optional[str]:
        """
        Create a new round in the rounds dataset.
        
        Args:
            round_number: The round number
            
        Returns:
            str: ID of the created round, or None if creation failed
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if rounds dataset exists and is properly initialized
            if "rounds" not in self.datasets or self.datasets["rounds"] is None:
                logger.info("Rounds dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Create a unique ID for this round
            round_id = str(uuid.uuid4())
            
            # Create a record
            record = rg.Record(
                fields={
                    "number": str(round_number),
                    "status": "in_progress"
                },
                metadata={
                    "round_id": round_id,
                    "start_time": datetime.now().isoformat(),
                    "end_time": "",
                    "metrics": "{}"
                }
            )
            
            # Add the record to the dataset
            try:
                self.datasets["rounds"].records.log([record])
                logger.info(f"Created round with ID: {round_id}")
                return round_id
            except Exception as log_error:
                logger.error(f"Error logging round record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry round creation")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["rounds"].records.log([record])
                        logger.info(f"Successfully created round after retry")
                        return round_id
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error creating round: {str(e)}")
            return None
    
    def update_round(self, round_id: str, status: str = None, 
                   metrics: Dict[str, Any] = None) -> bool:
        """
        Update a round with status and metrics.
        
        Args:
            round_id: ID of the round to update
            status: Status of the round ('in_progress' or 'completed')
            metrics: Dictionary of metrics for the round
            
        Returns:
            bool: True if update is successful, False otherwise
        """
        try:
            if not self.client or "rounds" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return False
            
            # Get the round record
            round_filter = rg.Filter(("metadata.round_id", "==", round_id))
            query = rg.Query(filter=round_filter)
            records = self.datasets["rounds"].records(query=query).to_list(flatten=True)
            
            if not records:
                logger.error(f"Round with ID {round_id} not found")
                return False
            
            # Prepare updates
            updates = {}
            
            if status:
                updates["status"] = status
            
            metadata_updates = {}
            
            if status == "completed":
                metadata_updates["end_time"] = datetime.now().isoformat()
            
            if metrics:
                metadata_updates["metrics"] = json.dumps(metrics)
            
            # Update the record
            record = records[0]
            updated_record = rg.Record(
                id=record["id"]
            )
            
            if updates:
                updated_record.fields = updates
                
            if metadata_updates:
                updated_record.metadata = metadata_updates
            
            # Add the updated record to the dataset
            try:
                self.datasets["rounds"].records.log([updated_record])
                logger.info(f"Updated round with ID: {round_id}")
                return True
            except Exception as log_error:
                logger.error(f"Error updating round: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["rounds"].records.log([updated_record])
                        logger.info(f"Successfully updated round after retry")
                        return True
                    except Exception as retry_error:
                        logger.error(f"Retry update failed: {str(retry_error)}")
                
                return False
            
        except Exception as e:
            logger.error(f"Error updating round: {str(e)}")
            return False
    
    def store_prompt_lineage(self, parent_prompt_id: str, child_prompt_id: str, 
                           round_id: str, change_reason: str) -> Optional[str]:
        """
        Store prompt lineage information in the prompt_lineage dataset.
        
        Args:
            parent_prompt_id: ID of the parent prompt
            child_prompt_id: ID of the child prompt
            round_id: ID of the round in which the change occurred
            change_reason: Reason for the change
            
        Returns:
            str: ID of the stored lineage record, or None if storage failed
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if prompt_lineage dataset exists and is properly initialized
            if "prompt_lineage" not in self.datasets or self.datasets["prompt_lineage"] is None:
                logger.info("Prompt lineage dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Create a unique ID for this lineage record
            lineage_id = str(uuid.uuid4())
            
            # Create a record
            record = rg.Record(
                fields={
                    "change_reason": change_reason
                },
                metadata={
                    "lineage_id": lineage_id,
                    "parent_prompt_id": parent_prompt_id,
                    "child_prompt_id": child_prompt_id,
                    "round_id": round_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Add the record to the dataset
            try:
                self.datasets["prompt_lineage"].records.log([record])
                logger.info(f"Stored prompt lineage with ID: {lineage_id}")
                return lineage_id
            except Exception as log_error:
                logger.error(f"Error logging prompt lineage record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry prompt lineage logging")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["prompt_lineage"].records.log([record])
                        logger.info(f"Successfully stored prompt lineage after retry")
                        return lineage_id
                    except Exception as retry_error:
                        logger.error(f"Retry logging failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error storing prompt lineage: {str(e)}")
            return None
    
    def _get_record(self, record_id: str) -> Optional[Dict]:
        """
        Helper method to get a record by ID.
        
        Args:
            record_id: ID of the record
            
        Returns:
            Dict: Record data, or None if not found
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if records dataset exists and is properly initialized
            if "records" not in self.datasets or self.datasets["records"] is None:
                logger.info("Records dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
                
            # Confirm that we actually have the dataset now
            if self.datasets["records"] is None:
                logger.error("Records dataset still not available after initialization")
                return None
            
            # Query for the record by ID
            try:
                record = self.datasets["records"].record(record_id)
                if record is None:
                    logger.warning(f"No record found with ID: {record_id}")
                    return None
                    
                # Return the record data
                return record
            except Exception as query_error:
                logger.error(f"Error querying for record: {str(query_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry record query")
                if self.connect() and self.initialize_datasets():
                    try:
                        record = self.datasets["records"].record(record_id)
                        return record
                    except Exception as retry_error:
                        logger.error(f"Retry query failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error getting record: {str(e)}")
            return None
    
    def _get_generator_prompt(self, prompt_id: str) -> Optional[Dict]:
        """
        Helper method to get a generator prompt by ID.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Dict: Prompt data, or None if not found
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if generator_prompts dataset exists and is properly initialized
            if "generator_prompts" not in self.datasets or self.datasets["generator_prompts"] is None:
                logger.info("Generator prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Query for the prompt using Filter
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            
            try:
                records = self.datasets["generator_prompts"].records(query=query).to_list(flatten=True)
                
                if not records:
                    logger.warning(f"No generator prompt found with ID: {prompt_id}")
                    return None
                    
                # Return the first matching record
                return records[0]
            except Exception as query_error:
                logger.error(f"Error querying for generator prompt: {str(query_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry generator prompt query")
                if self.connect() and self.initialize_datasets():
                    try:
                        records = self.datasets["generator_prompts"].records(query=query).to_list(flatten=True)
                        return records[0] if records else None
                    except Exception as retry_error:
                        logger.error(f"Retry query failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error getting generator prompt: {str(e)}")
            return None
    
    def _get_evaluator_prompt(self, prompt_id: str) -> Optional[Dict]:
        """
        Helper method to get an evaluator prompt by ID.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            Dict: Prompt data, or None if not found
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if evaluator_prompts dataset exists and is properly initialized
            if "evaluator_prompts" not in self.datasets or self.datasets["evaluator_prompts"] is None:
                logger.info("Evaluator prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Query for the prompt using Filter
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            
            try:
                records = self.datasets["evaluator_prompts"].records(query=query).to_list(flatten=True)
                
                if not records:
                    logger.warning(f"No evaluator prompt found with ID: {prompt_id}")
                    return None
                    
                # Return the first matching record
                return records[0]
            except Exception as query_error:
                logger.error(f"Error querying for evaluator prompt: {str(query_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry evaluator prompt query")
                if self.connect() and self.initialize_datasets():
                    try:
                        records = self.datasets["evaluator_prompts"].records(query=query).to_list(flatten=True)
                        return records[0] if records else None
                    except Exception as retry_error:
                        logger.error(f"Retry query failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error getting evaluator prompt: {str(e)}")
            return None
    
    def _get_round(self, round_id: str) -> Optional[Dict]:
        """
        Helper method to get a round by ID.
        
        Args:
            round_id: ID of the round
            
        Returns:
            Dict: Round data, or None if not found
        """
        try:
            # First ensure we're connected to Argilla
            if not self.client:
                logger.info("No client connection, attempting to connect")
                success = self.connect()
                if not success:
                    logger.error("Failed to connect to Argilla server")
                    return None
            
            # Check if rounds dataset exists and is properly initialized
            if "rounds" not in self.datasets or self.datasets["rounds"] is None:
                logger.info("Rounds dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
            
            # Query for the round using Filter
            round_filter = rg.Filter(("metadata.round_id", "==", round_id))
            query = rg.Query(filter=round_filter)
            
            try:
                records = self.datasets["rounds"].records(query=query).to_list(flatten=True)
                
                if not records:
                    logger.warning(f"No round found with ID: {round_id}")
                    return None
                    
                # Return the first matching record
                return records[0]
            except Exception as query_error:
                logger.error(f"Error querying for round: {str(query_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry round query")
                if self.connect() and self.initialize_datasets():
                    try:
                        records = self.datasets["rounds"].records(query=query).to_list(flatten=True)
                        return records[0] if records else None
                    except Exception as retry_error:
                        logger.error(f"Retry query failed: {str(retry_error)}")
                
                return None
            
        except Exception as e:
            logger.error(f"Error getting round: {str(e)}")
            return None
    
    def get_active_evaluator_prompts(self) -> List[Dict[str, Any]]:
        """
        Get the current active evaluator prompts.
        
        Returns:
            List[Dict[str, Any]]: List of prompt dictionaries
        """
        try:
            if not self.client or "evaluator_prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return []
            
            # Query for active prompts
            active_filter = rg.Filter(("metadata.active_status", "==", "true"))  # Updated to match renamed property
            query = rg.Query(filter=active_filter)
            active_prompts = self.datasets["evaluator_prompts"].records(query=query).to_list(flatten=True)
            
            # Extract prompt data
            prompt_data = []
            for prompt in active_prompts:
                prompt_data.append({
                    "id": prompt["metadata"].get("prompt_id", "unknown"),
                    "content": prompt["fields"]["content"],
                    "version": int(prompt["metadata"].get("version", 1)),
                    "created_at": prompt["metadata"].get("created_at", datetime.now().isoformat())
                })
            
            logger.info(f"Retrieved {len(prompt_data)} active evaluator prompts")
            return prompt_data
            
        except Exception as e:
            logger.error(f"Error getting active evaluator prompts: {str(e)}")
            return []
    
    def get_round_metrics(self, round_id: str) -> Dict[str, Any]:
        """
        Get metrics for a specific round.
        
        Args:
            round_id: ID of the round
            
        Returns:
            Dict[str, Any]: Dictionary containing round metrics
        """
        try:
            if not self.client or "rounds" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return {}
            
            # Query for the round
            round_filter = rg.Filter(("metadata.round_id", "==", round_id))
            query = rg.Query(filter=round_filter)
            rounds = self.datasets["rounds"].records(query=query).to_list(flatten=True)
            
            if not rounds:
                logger.warning(f"Round with ID {round_id} not found")
                return {}
            
            # Get the metrics from the round's metadata
            round_data = rounds[0]
            metrics_json = round_data["metadata"].get("metrics", "{}")
            
            try:
                metrics = json.loads(metrics_json)
            except json.JSONDecodeError:
                logger.error(f"Error decoding metrics JSON for round {round_id}")
                metrics = {}
            
            logger.info(f"Retrieved metrics for round ID: {round_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting round metrics: {str(e)}")
            return {}
    
    def get_active_generator_prompts(self, top_n: int = 4) -> List[Dict[str, Any]]:
        """
        Get the current active generator prompts.
        
        Args:
            top_n: Number of top prompts to return
            
        Returns:
            List[Dict[str, Any]]: List of prompt dictionaries
        """
        try:
            if not self.client or "generator_prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return []
            
            # Query for active prompts
            active_filter = rg.Filter(("metadata.active_status", "==", "true"))  # Updated to match renamed property
            query = rg.Query(filter=active_filter)
            active_prompts = self.datasets["generator_prompts"].records(query=query).to_list(flatten=True)
            
            # Extract prompt data
            prompt_data = []
            for prompt in active_prompts:
                prompt_data.append({
                    "id": prompt["metadata"].get("prompt_id", "unknown"),
                    "content": prompt["fields"]["content"],
                    "average_score": float(prompt["metadata"].get("avg_score_metadata", 0)),
                    "rounds_survived": int(prompt["metadata"].get("rounds_survived", 0)),
                    "version": int(prompt["metadata"].get("version", 1)),
                    "parent_prompt_id": prompt["metadata"].get("parent_prompt_id", "root")
                })
            
            # Sort by average score (descending) and take top N
            prompt_data.sort(key=lambda x: x["average_score"], reverse=True)
            top_prompts = prompt_data[:top_n]
            
            logger.info(f"Retrieved top {len(top_prompts)} active generator prompts")
            return top_prompts
            
        except Exception as e:
            logger.error(f"Error getting active generator prompts: {str(e)}")
            return []
    
    def get_prompt_lineage(self) -> Optional[pd.DataFrame]:
        """
        Get the lineage information for prompts.
        
        Returns:
            pd.DataFrame: DataFrame containing prompt lineage information
        """
        try:
            if not self.client or "prompt_lineage" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Get all lineage records
            all_lineage = self.datasets["prompt_lineage"].records().to_list(flatten=True)
            
            rows = []
            for lineage in all_lineage:
                row = {
                    "lineage_id": lineage["metadata"].get("lineage_id", "unknown"),
                    "parent_prompt_id": lineage["metadata"].get("parent_prompt_id", "unknown"),
                    "child_prompt_id": lineage["metadata"].get("child_prompt_id", "unknown"),
                    "round_id": lineage["metadata"].get("round_id", "unknown"),
                    "change_reason": lineage["fields"]["change_reason"],
                    "timestamp": lineage["metadata"].get("timestamp", datetime.now().isoformat())
                }
                rows.append(row)
            
            # Create the DataFrame
            df = pd.DataFrame(rows)
            
            # Sort by timestamp
            if not df.empty:
                df = df.sort_values(by=["timestamp"])
            
            logger.info(f"Retrieved prompt lineage with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error getting prompt lineage: {str(e)}")
            return None
    
    def get_prompts_from_records(self, prompt_type: str = "generator", 
                            generations: List[int] = None, 
                            top_n: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve prompts directly from the records dataset.
        
        Args:
            prompt_type: Type of prompt to retrieve ("generator" or "evaluator")
            generations: List of specific generations to filter by
            top_n: Number of top performing prompts to return
        
        Returns:
            List of dictionaries containing prompt information
        """
        try:
            if not self.client or "records" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return []
            
            # Determine which field to extract based on prompt type
            prompt_field = "generator_prompt" if prompt_type == "generator" else "evaluator_prompt"
            
            # Prepare search filters based on generations
            filters = {}
            if generations:
                # Convert generations to strings for metadata matching
                gen_strings = [str(g) for g in generations]
                filters["prompt_generation"] = {"$in": gen_strings}
            
            # Search for records
            results = self.datasets["records"].records.search(
                query="*",  # Match all records
                filters=filters,
                limit=100   # Get a reasonable number of records
            )
            
            # Process results to extract prompts, avoiding duplicates
            prompts_seen = set()
            prompts = []
            
            for record in results:
                try:
                    # Extract prompt content and metadata
                    prompt_content = record.fields.get(prompt_field, "")
                    
                    # Skip if we've already seen this prompt
                    if prompt_content in prompts_seen:
                        continue
                    
                    # Extract generation and other metadata
                    generation = int(record.metadata.get("prompt_generation", 0))
                    round_id = record.metadata.get("round_id", "")
                    timestamp = record.metadata.get("timestamp", "")
                    
                    # Calculate average score from questions if available
                    score = 0.0
                    if hasattr(record, 'questions') and 'ai_score' in record.questions:
                        score = float(record.questions.get('ai_score', 0))
                    
                    # Add to list of unique prompts
                    prompts_seen.add(prompt_content)
                    prompts.append({
                        'content': prompt_content,
                        'generation': generation,
                        'score': score,
                        'round_id': round_id,
                        'timestamp': timestamp
                    })
                except Exception as e:
                    logger.error(f"Error processing record: {str(e)}")
                    continue
            
            # Sort prompts by score (descending) and limit to top_n
            prompts.sort(key=lambda x: x['score'], reverse=True)
            return prompts[:top_n]
            
        except Exception as e:
            logger.error(f"Error retrieving prompts from records: {str(e)}")
            return []
