import argilla as rg
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArgillaDatabase:
    """
    Handles all interactions with the Argilla database for the Archer system.
    Responsible for storing and retrieving data related to prompts, generated content,
    evaluations, and performance metrics.
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
        Initialize all required datasets in Argilla.
        Creates the datasets if they don't exist.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        try:
            if not self.client:
                success = self.connect()
                if not success:
                    return False
            
            # Initialize Outputs Dataset
            try:
                # First check if dataset exists by trying to fetch it
                try:
                    outputs_dataset = self.client.datasets("archer_outputs")
                    # Verify the dataset was actually found and not None
                    if outputs_dataset is not None:
                        self.datasets["outputs"] = outputs_dataset
                        logger.info("Found existing outputs dataset")
                    else:
                        raise Exception("Dataset returned is None")
                except Exception as e:
                    logger.info(f"Creating new outputs dataset: {str(e)}")
                    outputs_settings = rg.Settings(
                        fields=[
                            rg.TextField(name="input", title="Input Data"),
                            rg.TextField(name="generated_content", title="Generated Content"),
                            rg.TextField(name="prompt_used", title="Prompt Used")
                        ],
                        questions=[
                            rg.RatingQuestion(
                                name="score",
                                title="Quality Score",
                                description="Rate the quality of the output",
                                values=[1, 2, 3, 4, 5]
                            ),
                            rg.TextQuestion(
                                name="feedback",
                                title="Feedback",
                                description="Provide feedback on how to improve the output"
                            ),
                        ],
                        metadata=[
                            rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                            rg.TermsMetadataProperty(name="round", title="Round Number"),
                            rg.TermsMetadataProperty(name="timestamp", title="Timestamp"),
                            rg.TermsMetadataProperty(name="output_id", title="Output ID")
                        ]
                    )
                    logger.info("Creating archer_outputs dataset...")
                    new_dataset = rg.Dataset(name="archer_outputs", settings=outputs_settings)
                    new_dataset.create()
                    # Verify creation worked by fetching again
                    self.datasets["outputs"] = self.client.datasets("archer_outputs")
                    if self.datasets["outputs"] is None:
                        raise Exception("Failed to create archer_outputs dataset")
                    logger.info("Created new outputs dataset")
            except Exception as e:
                logger.error(f"Error with outputs dataset: {str(e)}")
                return False
            
            # Initialize Prompts Dataset
            try:
                # First check if dataset exists by trying to fetch it
                try:
                    prompts_dataset = self.client.datasets("archer_prompts")
                    # Verify the dataset was actually found and not None
                    if prompts_dataset is not None:
                        self.datasets["prompts"] = prompts_dataset
                        logger.info("Found existing prompts dataset")
                    else:
                        raise Exception("Dataset returned is None")
                except Exception as e:
                    logger.info(f"Creating new prompts dataset: {str(e)}")
                    prompts_settings = rg.Settings(
                        fields=[
                            rg.TextField(name="prompt_text", title="Prompt Text"),
                            rg.TextField(name="model", title="Model"),
                            rg.TextField(name="purpose", title="Purpose")
                        ],
                        questions=[
                            rg.RatingQuestion(
                                name="average_score",
                                title="Average Score",
                                description="Average performance score for this prompt",
                                values=[1, 2, 3, 4, 5]
                            ),
                            rg.LabelQuestion(
                                name="survived",
                                title="Survived",
                                description="Whether this prompt survived to the next generation",
                                labels=["True", "False"]
                            )
                        ],
                        metadata=[
                            rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                            rg.TermsMetadataProperty(name="parent_prompt_id", title="Parent Prompt ID"),
                            rg.TermsMetadataProperty(name="generation", title="Generation Number"),
                            rg.TermsMetadataProperty(name="timestamp", title="Timestamp")
                        ]
                    )
                    logger.info("Creating archer_prompts dataset...")
                    new_dataset = rg.Dataset(name="archer_prompts", settings=prompts_settings)
                    new_dataset.create()
                    # Verify creation worked by fetching again
                    self.datasets["prompts"] = self.client.datasets("archer_prompts")
                    if self.datasets["prompts"] is None:
                        raise Exception("Failed to create archer_prompts dataset")
                    logger.info("Created new prompts dataset")
            except Exception as e:
                logger.error(f"Error with prompts dataset: {str(e)}")
                return False
            
            # Initialize Evaluations Dataset
            try:
                # First check if dataset exists by trying to fetch it
                try:
                    evaluations_dataset = self.client.datasets("archer_evaluations")
                    # Verify the dataset was actually found and not None
                    if evaluations_dataset is not None:
                        self.datasets["evaluations"] = evaluations_dataset
                        logger.info("Found existing evaluations dataset")
                    else:
                        raise Exception("Dataset returned is None")
                except Exception as e:
                    logger.info(f"Creating new evaluations dataset: {str(e)}")
                    evaluations_settings = rg.Settings(
                        fields=[
                            rg.TextField(name="input", title="Input Data"),
                            rg.TextField(name="generated_content", title="Generated Content"),
                            rg.TextField(name="evaluation_content", title="Evaluation")
                        ],
                        questions=[
                            rg.RatingQuestion(
                                name="score",
                                title="Quality Score",
                                description="Rate the quality of the output",
                                values=[1, 2, 3, 4, 5]
                            ),
                            rg.TextQuestion(
                                name="feedback",
                                title="Feedback",
                                description="Feedback on how to improve"
                            ),
                            rg.TextField(name="improved_output", title="Improved Output")
                        ],
                        metadata=[
                            rg.TermsMetadataProperty(name="output_id", title="Output ID"),
                            rg.TermsMetadataProperty(name="prompt_id", title="Prompt ID"),
                            rg.TermsMetadataProperty(name="evaluator_id", title="Evaluator ID"),
                            rg.TermsMetadataProperty(name="is_human", title="Is Human Evaluation"),
                            rg.TermsMetadataProperty(name="timestamp", title="Timestamp")
                        ]
                    )
                    logger.info("Creating archer_evaluations dataset...")
                    new_dataset = rg.Dataset(name="archer_evaluations", settings=evaluations_settings)
                    new_dataset.create()
                    # Verify creation worked by fetching again
                    self.datasets["evaluations"] = self.client.datasets("archer_evaluations")
                    if self.datasets["evaluations"] is None:
                        raise Exception("Failed to create archer_evaluations dataset")
                    logger.info("Created new evaluations dataset")
            except Exception as e:
                logger.error(f"Error with evaluations dataset: {str(e)}")
                return False
                
            # Verify all datasets exist
            if not all(self.datasets.get(k) is not None for k in ["outputs", "prompts", "evaluations"]):
                logger.error("Not all datasets were properly initialized")
                missing = [k for k in ["outputs", "prompts", "evaluations"] if self.datasets.get(k) is None]
                logger.error(f"Missing datasets: {', '.join(missing)}")
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
            
            # Handle the output structure - extract fields and metadata safely
            try:
                # Safely extract fields data
                if isinstance(output, dict):
                    if 'fields' in output:
                        input_data = output["fields"].get("input", "")
                        generated_content = output["fields"].get("generated_content", "")
                        prompt_id = output["metadata"].get("prompt_id", "unknown") if "metadata" in output else "unknown"
                    elif "input" in output and "generated_content" in output:
                        # Alternative structure: keys are at the top level
                        input_data = output.get("input", "")
                        generated_content = output.get("generated_content", "")
                        # Check if metadata is a nested structure or at top level
                        if "metadata" in output and isinstance(output["metadata"], dict):
                            prompt_id = output["metadata"].get("prompt_id", "unknown")
                        else:
                            prompt_id = output.get("prompt_id", "unknown")
                    else:
                        logger.error(f"Output found but has unexpected dictionary structure: {list(output.keys())}")
                        return False
                elif hasattr(output, 'fields'):
                    input_data = getattr(getattr(output, 'fields', object()), 'input', "")
                    generated_content = getattr(getattr(output, 'fields', object()), 'generated_content', "")
                    prompt_id = getattr(getattr(output, 'metadata', object()), 'prompt_id', "unknown")
                else:
                    logger.error(f"Output found but has unexpected structure: {type(output)}")
                    return False
            except Exception as e:
                logger.error(f"Error extracting data from output: {str(e)}")
                return False
            
            # Create a unique ID for this evaluation
            evaluator_id = "human" if is_human else "ai_evaluator"
            
            # Create responses
            responses = [
                rg.Response(question_name="score", value=score, user_id=self.user_id),
                rg.Response(question_name="feedback", value=feedback, user_id=self.user_id),
                rg.Response(question_name="improved_output", value=improved_output, user_id=self.user_id)
            ]
            
            # Create a record
            record = rg.Record(
                fields={
                    "input": input_data,
                    "generated_content": generated_content,
                    "evaluation_content": ""  # This will be filled by responses
                },
                responses=responses,
                metadata={
                    "output_id": output_id,
                    "prompt_id": prompt_id,
                    "evaluator_id": evaluator_id,
                    "is_human": "1" if is_human else "0",
                    "timestamp": datetime.now().isoformat()
                }
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
            
            # Check if prompts dataset exists and is properly initialized
            if "prompts" not in self.datasets or self.datasets["prompts"] is None:
                logger.info("Prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
                
            # Confirm that we actually have the dataset now
            if self.datasets["prompts"] is None:
                logger.error("Prompts dataset still not available after initialization")
                return None
            
            # Create a unique ID for this prompt
            prompt_id = str(uuid.uuid4())
            
            # Create responses
            responses = [
                rg.Response(question_name="average_score", value=0.0, user_id=self.user_id),
                rg.Response(question_name="survived", value=False, user_id=self.user_id)
            ]
            
            # Create a record
            record = rg.Record(
                fields={
                    "prompt_text": prompt_text,
                    "model": model,
                    "purpose": purpose
                },
                responses=responses,
                metadata={
                    "prompt_id": prompt_id,
                    "parent_prompt_id": parent_prompt_id or "root",
                    "generation": str(generation),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            try:
                # Add the record to the dataset
                self.datasets["prompts"].records.log([record])
                logger.info(f"Stored prompt with ID: {prompt_id}")
                return prompt_id
            except Exception as log_error:
                logger.error(f"Error logging prompt record: {str(log_error)}")
                
                # Try to reconnect and retry
                logger.info("Attempting to reconnect and retry prompt logging")
                if self.connect() and self.initialize_datasets():
                    try:
                        self.datasets["prompts"].records.log([record])
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
            if not self.client or "prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return False
            
            # Get the prompt record
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            records = self.datasets["prompts"].records(query=query).to_list(flatten=True)
            
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
            
            # Add the updated record to the dataset
            self.datasets["prompts"].records.log([updated_record])
            
            logger.info(f"Updated performance for prompt ID: {prompt_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prompt performance: {str(e)}")
            return False
    
    def get_current_data_for_annotation(self, round_num: int, limit: int = 20) -> Optional[pd.DataFrame]:
        """
        Get the current data for human annotation.
        
        Args:
            round_num: Current round number
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
            if not all(k in self.datasets for k in ["outputs", "evaluations"]):
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Query for outputs from the current round
            round_filter = rg.Filter(("metadata.round", "==", str(round_num)))
            query = rg.Query(filter=round_filter)
            outputs = self.datasets["outputs"].records(query=query).to_list(flatten=True)
            
            # Limit the number of records
            outputs = outputs[:limit]
            
            if not outputs:
                logger.warning(f"No outputs found for round {round_num}")
                return pd.DataFrame()
            
            # Build a DataFrame with the required columns
            rows = []
            for output in outputs:
                # Get the latest evaluation for this output
                output_id = output["metadata"]["output_id"]
                evaluation = self._get_latest_evaluation(output_id)
                
                row = {
                    "output_id": output_id,
                    "input": output["fields"]["input"],
                    "eval_content": output["fields"]["generated_content"],
                    "eval_score": evaluation.get("score", 0) if evaluation else 0,
                    "eval_feedback": evaluation.get("feedback", "") if evaluation else "",
                    "eval_perfect_output": evaluation.get("improved_output", "") if evaluation else "",
                    "prompt_id": output["metadata"]["prompt_id"]
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
                    return {}
                    
            # Ensure datasets are initialized
            if not all(k in self.datasets for k in ["outputs", "evaluations", "prompts"]):
                success = self.initialize_datasets()
                if not success:
                    return {}
            
            # Get all outputs with evaluations
            all_records = self.datasets["outputs"].records().to_list(flatten=True)
            
            # Get all evaluations
            all_evaluations = self.datasets["evaluations"].records().to_list(flatten=True)
            
            # Get all prompts
            all_prompts = self.datasets["prompts"].records().to_list(flatten=True)
            
            # Calculate metrics
            metrics = {
                "rounds": [],
                "prompts": [],
                "scores": [],
                "prompt_survivorship": {}
            }
            
            # Process prompts
            for prompt in all_prompts:
                prompt_id = prompt["metadata"]["prompt_id"]
                generation = int(prompt["metadata"]["generation"])
                parent_id = prompt["metadata"]["parent_prompt_id"]
                
                # Extract responses
                avg_score = 0
                survived = False
                if "responses" in prompt:
                    for response in prompt["responses"]:
                        if response["question"]["name"] == "average_score":
                            avg_score = float(response["value"])
                        elif response["question"]["name"] == "survived":
                            survived = bool(response["value"])
                
                # Track prompt metrics
                metrics["prompts"].append({
                    "id": prompt_id,
                    "generation": generation,
                    "parent_id": parent_id,
                    "avg_score": avg_score,
                    "survived": survived
                })
                
                # Track survivorship
                if prompt_id not in metrics["prompt_survivorship"]:
                    metrics["prompt_survivorship"][prompt_id] = {
                        "generations": [generation],
                        "scores": [avg_score]
                    }
                else:
                    metrics["prompt_survivorship"][prompt_id]["generations"].append(generation)
                    metrics["prompt_survivorship"][prompt_id]["scores"].append(avg_score)
            
            # Process outputs and evaluations
            for output in all_records:
                output_id = output["metadata"]["output_id"]
                round_num = int(output["metadata"]["round"])
                prompt_id = output["metadata"]["prompt_id"]
                
                # Find evaluations for this output
                output_evaluations = [e for e in all_evaluations if e["metadata"]["output_id"] == output_id]
                if output_evaluations:
                    # Use the latest evaluation
                    latest_eval = max(output_evaluations, key=lambda e: e["metadata"]["timestamp"])
                    
                    # Extract score from responses
                    score = 0
                    if "responses" in latest_eval:
                        for response in latest_eval["responses"]:
                            if response["question"]["name"] == "score":
                                score = int(response["value"])
                                break
                    
                    metrics["rounds"].append(round_num)
                    metrics["scores"].append(score)
            
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
            return {}
    
    def get_prompt_history(self) -> Optional[pd.DataFrame]:
        """
        Get the history of prompts across generations.
        
        Returns:
            pd.DataFrame: DataFrame containing prompt history
        """
        try:
            if not self.client or "prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return None
            
            # Get all prompts
            all_prompts = self.datasets["prompts"].records().to_list(flatten=True)
            
            rows = []
            for prompt in all_prompts:
                # Extract responses
                avg_score = 0
                survived = False
                if "responses" in prompt:
                    for response in prompt["responses"]:
                        if response["question"]["name"] == "average_score":
                            avg_score = float(response["value"])
                        elif response["question"]["name"] == "survived":
                            survived = bool(response["value"])
                
                row = {
                    "prompt_id": prompt["metadata"]["prompt_id"],
                    "generation": int(prompt["metadata"]["generation"]),
                    "parent_id": prompt["metadata"].get("parent_prompt_id", "root"),
                    "prompt_text": prompt["fields"]["prompt_text"],
                    "model": prompt["fields"]["model"],
                    "purpose": prompt["fields"]["purpose"],
                    "avg_score": avg_score,
                    "survived": survived,
                    "timestamp": prompt["metadata"]["timestamp"]
                }
                rows.append(row)
            
            # Create the DataFrame
            df = pd.DataFrame(rows)
            
            # Sort by generation and timestamp
            if not df.empty:
                df = df.sort_values(by=["generation", "timestamp"])
            
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
            if not self.client or "prompts" not in self.datasets:
                success = self.initialize_datasets()
                if not success:
                    return []
            
            # Get all prompts
            all_prompts = self.datasets["prompts"].records().to_list(flatten=True)
            
            # Extract prompt IDs and scores
            prompt_scores = []
            for prompt in all_prompts:
                prompt_id = prompt["metadata"]["prompt_id"]
                
                # Extract avg_score from responses
                avg_score = 0
                if "responses" in prompt:
                    for response in prompt["responses"]:
                        if response["question"]["name"] == "average_score":
                            avg_score = float(response["value"])
                            break
                
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

            # Check if prompts dataset exists and is properly initialized
            if "prompts" not in self.datasets or self.datasets["prompts"] is None:
                logger.info("Prompts dataset not loaded yet, initializing datasets")
                success = self.initialize_datasets()
                if not success:
                    logger.error("Failed to initialize datasets")
                    return None
                
            # Confirm that we actually have the dataset now
            if self.datasets["prompts"] is None:
                logger.error("Prompts dataset still not available after initialization")
                return None

            # Query for the prompt using the Filter API
            prompt_filter = rg.Filter(("metadata.prompt_id", "==", prompt_id))
            query = rg.Query(filter=prompt_filter)
            
            try:
                records_iterator = self.datasets["prompts"].records(query=query)
                records_list = records_iterator.to_list(flatten=True)
                
                if not records_list:
                    logger.warning(f"No prompt found with ID: {prompt_id}")
                    return None
                    
                # Return the prompt text - handle possible data structure differences
                record = records_list[0]
                if isinstance(record, dict) and "fields" in record and "prompt_text" in record["fields"]:
                    return record["fields"]["prompt_text"]
                elif hasattr(record, 'fields') and hasattr(record.fields, 'prompt_text'):
                    return record.fields.prompt_text
                else:
                    logger.warning(f"Prompt found but could not extract text, record structure: {record}")
                    return None
            except Exception as query_error:
                logger.error(f"Error querying for prompt: {str(query_error)}")
                
                # Try once more with a fresh connection
                logger.info("Attempting to reconnect and retry prompt query")
                if self.connect() and self.initialize_datasets():
                    try:
                        records_iterator = self.datasets["prompts"].records(query=query)
                        records_list = records_iterator.to_list(flatten=True)
                        if records_list:
                            record = records_list[0]
                            if isinstance(record, dict) and "fields" in record and "prompt_text" in record["fields"]:
                                return record["fields"]["prompt_text"]
                            elif hasattr(record, 'fields') and hasattr(record.fields, 'prompt_text'):
                                return record.fields.prompt_text
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
                        if response["question"]["name"] == "score":
                            evaluation["score"] = int(response["value"])
                        elif response["question"]["name"] == "feedback":
                            evaluation["feedback"] = response["value"]
                        elif response["question"]["name"] == "improved_output":
                            evaluation["improved_output"] = response["value"]
                elif hasattr(latest, 'responses'):
                    for response in latest.responses:
                        if hasattr(response, 'question') and hasattr(response.question, 'name'):
                            if response.question.name == "score":
                                evaluation["score"] = int(response.value)
                            elif response.question.name == "feedback":
                                evaluation["feedback"] = response.value
                            elif response.question.name == "improved_output":
                                evaluation["improved_output"] = response.value
                
                return evaluation
                
            except Exception as query_error:
                logger.error(f"Error querying for evaluations: {str(query_error)}")
                return None
            
        except Exception as e:
            logger.error(f"Error getting latest evaluation: {str(e)}")
            return None
