import os
import logging
import uuid
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from supabase import create_client, Client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SupabaseDatabase:
    """
    Handles all interactions with the Supabase database for the Archer system.

    This class implements the revised schema using SQL on Supabase.
    Key tables include:
      - archer_records: Main table for input data, generated content, and evaluations.
      - archer_prompts: Consolidated table for both generator and evaluator prompts.
      - archer_rounds: Tracks iteration/round information.
      - archer_prompt_lineage: Tracks evolution of prompts.
      - archer_outputs: Stores generated outputs.
      - archer_evaluations: Stores evaluations (AI and human) of generated outputs.
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the SupabaseDatabase instance using Supabase URL and API key.
        """
        self.api_url = api_url or os.getenv("SUPABASE_API_URL")
        self.api_key = api_key or os.getenv("SUPABASE_API_KEY")
        self.client: Client = create_client(self.api_url, self.api_key)
        self.user_id = "default_user"
        self.datasets = {}

    def _safe_execute(self, query, operation_name="operation"):
        """
        Safely execute a Supabase query and handle any errors.
        
        Args:
            query: The Supabase query to execute
            operation_name: Name of the operation for logging
            
        Returns:
            tuple: (success, data) where success is a boolean and data is the result or None
        """
        try:
            result = query.execute()
            return True, result.data
        except Exception as e:
            logger.error(f"Error during {operation_name}: {str(e)}")
            return False, None

    def connect(self) -> bool:
        """
        Connect to the Supabase database.
        This is a compatibility method for code that was previously using Argilla.
        Since Supabase client is initialized in __init__, this method just verifies
        the connection and returns True if it's working.
        """
        try:
            # Try a simple query to verify the connection works
            response = self.client.from_("archer_records").select("count", count="exact").limit(1).execute()
            # Supabase responses don't have an error property like Argilla did
            # If the query executed without an exception, we're good
            logger.info("Successfully connected to Supabase database")
            return True
        except Exception as e:
            logger.error(f"Exception while connecting to Supabase: {str(e)}")
            return False

    def initialize_datasets(self) -> bool:
        """
        Initialize the required datasets in Supabase.
        This is a compatibility method for code that was previously using Argilla.
        In Supabase, tables should already be created in the database schema,
        so this method verifies their existence and initializes dataset references.
        
        Returns:
            bool: True if all datasets/tables exist and are accessible, False otherwise.
        """
        try:
            # Verify all required tables exist by attempting to select from them
            tables = [
                "archer_records",
                "archer_prompts",
                "archer_rounds",
                "archer_prompt_lineage",
                "archer_outputs",
                "archer_evaluations"
            ]
            
            for table in tables:
                try:
                    # Use try/except here since Supabase responses don't have an error property
                    self.client.from_(table).select("count", count="exact").limit(1).execute()
                except Exception as e:
                    logger.error(f"Error accessing table {table}: {str(e)}")
                    return False
                
            # Initialize dataset references for backward compatibility
            self.datasets = {
                "records": {"name": "archer_records"},
                "generator_prompts": {"name": "archer_prompts"},
                "evaluator_prompts": {"name": "archer_prompts"},
                "rounds": {"name": "archer_rounds"},
                "prompt_lineage": {"name": "archer_prompt_lineage"},
                "outputs": {"name": "archer_outputs"},
                "evaluations": {"name": "archer_evaluations"}
            }
            
            logger.info("Successfully initialized all datasets/tables")
            return True
        except Exception as e:
            logger.error(f"Exception in initialize_datasets: {str(e)}")
            return False
            
    def _initialize_records_dataset(self) -> bool:
        """
        Helper method for backward compatibility.
        In Supabase, this is a no-op since tables are created via migrations.
        """
        return True
        
    def _initialize_generator_prompts_dataset(self) -> bool:
        """
        Helper method for backward compatibility.
        In Supabase, this is a no-op since tables are created via migrations.
        """
        return True
        
    def _initialize_evaluator_prompts_dataset(self) -> bool:
        """
        Helper method for backward compatibility.
        In Supabase, this is a no-op since tables are created via migrations.
        """
        return True
        
    def _initialize_rounds_dataset(self) -> bool:
        """
        Helper method for backward compatibility.
        In Supabase, this is a no-op since tables are created via migrations.
        """
        return True
        
    def _initialize_prompt_lineage_dataset(self) -> bool:
        """
        Helper method for backward compatibility.
        In Supabase, this is a no-op since tables are created via migrations.
        """
        return True

    def store_generated_content(self, input_data: str, content: str, prompt_id: str, round_num: int) -> Optional[str]:
        """
        Store generated content in the archer_outputs table.
        
        Args:
            input_data: The input data used for generation
            content: The generated content
            prompt_id: ID of the prompt used for generation (must be a valid ID from archer_prompts)
            round_num: The round number
            
        Returns:
            The output ID if successful, None otherwise
        """
        try:
            # Verify the prompt ID exists in the prompts table
            prompt_exists = False
            try:
                response = self.client.table("archer_prompts").select("id").eq("id", prompt_id).execute()
                prompt_exists = response and hasattr(response, 'data') and len(response.data) > 0
            except Exception as e:
                logger.warning(f"Failed to verify if prompt ID exists: {prompt_id}, error: {str(e)}")
            
            # If prompt doesn't exist, try to fetch it or create a new one
            if not prompt_exists:
                logger.warning(f"Prompt ID {prompt_id} does not exist in prompts table. Attempting to create it.")
                
                # Create a default prompt entry
                try:
                    # Use a placeholder content to avoid nulls
                    placeholder_content = "Generated prompt placeholder"
                    
                    # Create a new prompt with the given ID if possible
                    data = {
                        "id": prompt_id,
                        "content": placeholder_content,
                        "prompt_type": "generator",
                        "version": 1,
                        "average_score": 0.0,
                        "rounds_survived": 1,
                        "is_active": True,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    response = self.client.table("archer_prompts").insert(data).execute()
                    if response and hasattr(response, 'data') and len(response.data) > 0:
                        logger.info(f"Created new prompt with ID: {prompt_id}")
                        prompt_exists = True
                    else:
                        # If we can't create a prompt with the provided ID, generate a new one
                        logger.warning(f"Could not create prompt with ID: {prompt_id}. Generating a new one.")
                        new_prompt_id = self.store_generator_prompt(
                            content=placeholder_content
                        )
                        if new_prompt_id:
                            logger.info(f"Created new prompt with generated ID: {new_prompt_id}")
                            prompt_id = new_prompt_id
                            prompt_exists = True
                        else:
                            logger.error("Failed to create a new prompt. Content may not be properly associated.")
                except Exception as e:
                    logger.error(f"Error creating prompt: {str(e)}")
            
            output_id = str(uuid.uuid4())
            data = {
                "id": output_id,
                "input_data": input_data,
                "generated_content": content,
                "prompt_id": prompt_id,
                "round_num": round_num,
                "created_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_outputs").insert(data).execute()
            # Supabase responses don't have an error property like Argilla
            # If we get here without an exception, the insert was successful
            logger.info(f"Stored generated content with ID: {output_id}")
            return output_id
        except Exception as e:
            logger.error(f"Exception in store_generated_content: {str(e)}")
            return None

    def store_evaluation(self, output_id: str, score: int, feedback: str, improved_output: str, is_human: bool = False) -> bool:
        """
        Store an evaluation for a given output in the archer_evaluations table.
        
        Args:
            output_id: ID of the output being evaluated
            score: Integer score for the evaluation
            feedback: Feedback text
            improved_output: Improved version of the output
            is_human: Whether this is a human evaluation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Retrieve the original output for reference
            output = self._get_output(output_id)
            if not output:
                logger.error(f"Output with ID {output_id} not found")
                return False

            # Get prompt ID from output and ensure it exists
            prompt_id = output.get("prompt_id", "")
            
            if not prompt_id:
                logger.warning(f"Output {output_id} is missing prompt_id. Attempting to find or create one.")
                
                # 1. Try to find a matching prompt by content in archer_records
                try:
                    prompt_content = output.get("generated_content", "")[:100]  # Use part of content as a signature
                    if prompt_content:
                        records = self.client.table("archer_records").select("generator_prompt_id").eq("generated_content", output.get("generated_content", "")).execute()
                        if records and hasattr(records, 'data') and records.data and records.data[0].get("generator_prompt_id"):
                            prompt_id = records.data[0].get("generator_prompt_id", "")
                            logger.info(f"Found prompt ID {prompt_id} by content matching in records")
                except Exception as e:
                    logger.error(f"Error finding prompt by content in records: {str(e)}")
                
                # 2. If still no prompt_id, try to find a matching prompt by content in archer_prompts
                if not prompt_id:
                    try:
                        # Use generated content as a signature to find matching prompts
                        content_signature = output.get("generated_content", "")[:50]
                        response = self.client.table("archer_prompts").select("id, content").execute()
                        
                        if response and hasattr(response, 'data') and response.data:
                            # Look for similar content
                            for prompt in response.data:
                                if content_signature in prompt.get("content", ""):
                                    prompt_id = prompt.get("id")
                                    logger.info(f"Found prompt ID {prompt_id} with similar content")
                                    break
                    except Exception as e:
                        logger.error(f"Error finding prompt by similar content: {str(e)}")
                
                # 3. If still no prompt_id, create a new prompt
                if not prompt_id:
                    logger.info("Creating new prompt since none found")
                    prompt_content = "Evaluation prompt created from output " + output_id
                    prompt_id = self.store_generator_prompt(content=prompt_content)
                    
                    if prompt_id:
                        logger.info(f"Created new prompt with ID: {prompt_id}")
                        
                        # Also update the output record with this prompt_id
                        try:
                            update_data = {"prompt_id": prompt_id}
                            self.client.table("archer_outputs").update(update_data).eq("id", output_id).execute()
                            logger.info(f"Updated output {output_id} with prompt_id {prompt_id}")
                        except Exception as e:
                            logger.error(f"Error updating output with new prompt_id: {str(e)}")
                    else:
                        logger.error("Failed to create a new prompt")
                        # Continue anyway with a fallback UUID
                        prompt_id = str(uuid.uuid4())
                        logger.warning(f"Using fallback UUID as prompt_id: {prompt_id}")
            
            # Verify the prompt exists in the database
            if prompt_id:
                try:
                    response = self.client.table("archer_prompts").select("id").eq("id", prompt_id).execute()
                    prompt_exists = response and hasattr(response, 'data') and len(response.data) > 0
                    
                    if not prompt_exists:
                        logger.warning(f"Prompt ID {prompt_id} not found in database. Creating it.")
                        # Create a placeholder prompt with this ID
                        placeholder_content = "Placeholder prompt created during evaluation storage"
                        data = {
                            "id": prompt_id,
                            "content": placeholder_content,
                            "prompt_type": "generator",
                            "version": 1,
                            "is_active": True,
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat()
                        }
                        self.client.table("archer_prompts").insert(data).execute()
                        logger.info(f"Created placeholder prompt with ID: {prompt_id}")
                except Exception as e:
                    logger.error(f"Error verifying prompt existence: {str(e)}")

            evaluation_id = str(uuid.uuid4())
            
            # Ensure score is an integer
            try:
                score_int = int(float(score))
            except (ValueError, TypeError):
                logger.error(f"Invalid score value: {score}. Must be convertible to integer.")
                return False
                
            data = {
                "id": evaluation_id,
                "input": output.get("input_data", ""),
                "generated_content": output.get("generated_content", ""),
                "evaluation_content": "",  # Can be expanded if needed
                "score": score_int,
                "feedback": feedback,
                "improved_output": improved_output,
                "output_id": output_id,
                "prompt_id": prompt_id,
                "evaluator_id": "human" if is_human else "ai_evaluator",
                "is_human": is_human,
                "timestamp": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            
            success, _ = self._safe_execute(
                self.client.table("archer_evaluations").insert(data),
                "storing evaluation"
            )
            
            if not success:
                return False
                
            logger.info(f"Stored evaluation for output ID: {output_id}")
            return True
        except Exception as e:
            logger.error(f"Exception in store_evaluation: {str(e)}")
            return False

    def store_human_feedback(self, output_id: str, score: int, feedback: str, improved_output: str) -> bool:
        """
        Wrapper to store human feedback (evaluation) for an output.
        """
        return self.store_evaluation(output_id, score, feedback, improved_output, is_human=True)

    def store_prompt(self, content: str, prompt_type: str, parent_prompt_id: Optional[str] = None, version: int = 1) -> Optional[str]:
        """
        Store a prompt (either generator or evaluator) in the consolidated archer_prompts table.
        """
        try:
            # Check if a prompt with the same content already exists
            response = self.client.from_("archer_prompts")\
                .select("id, content, prompt_type")\
                .eq("prompt_type", prompt_type)\
                .execute()
            
            # If response is successful, check for duplicates
            if response and hasattr(response, 'data'):
                existing_prompts = response.data or []
                for existing_prompt in existing_prompts:
                    if existing_prompt.get("content") == content:
                        # Prompt already exists, return its ID
                        prompt_id = existing_prompt.get('id')
                        logger.info(f"Found existing {prompt_type} prompt with matching content, reusing ID: {prompt_id}")
                        logger.debug(f"Duplicate prompt content: {content[:50]}...")
                        return prompt_id
            
            # No duplicate found, create a new prompt
            prompt_id = str(uuid.uuid4())
            data = {
                "id": prompt_id,
                "content": content,
                "prompt_type": prompt_type,  # 'generator' or 'evaluator'
                "parent_prompt_id": parent_prompt_id,
                "version": version,
                "average_score": None,
                "rounds_survived": None,
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_prompts").insert(data).execute()
            # Supabase responses don't have an error property like Argilla
            # If we get here without an exception, the insert was successful
            logger.info(f"Stored {prompt_type} prompt with ID: {prompt_id}")
            return prompt_id
        except Exception as e:
            logger.error(f"Exception in store_prompt: {str(e)}")
            return None

    def store_generator_prompt(self, content: str, parent_prompt_id: Optional[str] = None, version: int = 1) -> Optional[str]:
        """
        Store a generator prompt in the archer_prompts table.
        This is a wrapper for the consolidated store_prompt method.
        """
        return self.store_prompt(content, "generator", parent_prompt_id, version)
        
    def store_evaluator_prompt(self, content: str, parent_prompt_id: Optional[str] = None, version: int = 1) -> Optional[str]:
        """
        Store an evaluator prompt in the archer_prompts table.
        This is a wrapper for the consolidated store_prompt method.
        """
        return self.store_prompt(content, "evaluator", parent_prompt_id, version)

    def update_prompt_performance(self, prompt_id: str, avg_score: float, rounds_survived: int, is_active: bool) -> bool:
        """
        Update performance metrics for a prompt in the archer_prompts table.
        """
        try:
            data = {
                "average_score": avg_score,
                "rounds_survived": rounds_survived,
                "is_active": is_active,
                "updated_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_prompts").update(data).eq("id", prompt_id).execute()
            # Supabase responses don't have an error property like Argilla
            # If we get here without an exception, the update was successful
            logger.info(f"Updated performance for prompt ID: {prompt_id}")
            return True
        except Exception as e:
            logger.error(f"Exception in update_prompt_performance: {str(e)}")
            return False
            
    def update_generator_prompt_performance(self, prompt_id: str, avg_score: float, rounds_survived: int, is_active: bool) -> bool:
        """
        Update performance metrics for a generator prompt.
        This is a wrapper for the consolidated update_prompt_performance method.
        """
        # First verify this is a generator prompt
        prompt = self._get_generator_prompt(prompt_id)
        if not prompt:
            logger.error(f"No generator prompt found with ID: {prompt_id}")
            return False
        return self.update_prompt_performance(prompt_id, avg_score, rounds_survived, is_active)
        
    def update_evaluator_prompt_performance(self, prompt_id: str, avg_score: float, rounds_survived: int, is_active: bool) -> bool:
        """
        Update performance metrics for an evaluator prompt.
        This is a wrapper for the consolidated update_prompt_performance method.
        """
        # First verify this is an evaluator prompt
        prompt = self._get_evaluator_prompt(prompt_id)
        if not prompt:
            logger.error(f"No evaluator prompt found with ID: {prompt_id}")
            return False
        return self.update_prompt_performance(prompt_id, avg_score, rounds_survived, is_active)

    def get_current_data_for_annotation(self, round_id: str, limit: int = 20) -> Optional[pd.DataFrame]:
        """
        Retrieve current records for human annotation based on a round ID.
        """
        try:
            success, records = self._safe_execute(
                self.client.table("archer_records").select("*").eq("round_id", round_id),
                "fetching records for annotation"
            )
            
            if not success:
                return pd.DataFrame()
                
            records = records or []
            records = records[:limit]
            if not records:
                logger.warning(f"No records found for round {round_id}")
                return pd.DataFrame()
            rows = []
            for record in records:
                row = {
                    "record_id": record.get("id"),
                    "input": record.get("input"),
                    "content": record.get("generated_content"),
                    "ai_score": record.get("ai_score"),
                    "ai_feedback": record.get("ai_feedback"),
                    "ai_improved_output": record.get("ai_improved_output"),
                    "human_score": record.get("human_score"),
                    "human_feedback": record.get("human_feedback"),
                    "human_improved_output": record.get("human_improved_output"),
                    "generator_prompt_id": record.get("generator_prompt_id"),
                    "evaluator_prompt_id": record.get("evaluator_prompt_id"),
                    "is_validated": record.get("validated_status", False)
                }
                rows.append(row)
            df = pd.DataFrame(rows)
            logger.info(f"Retrieved {len(df)} records for annotation")
            return df
        except Exception as e:
            logger.error(f"Exception in get_current_data_for_annotation: {str(e)}")
            return None

    def get_performance_metrics(self, max_rounds: int = 2) -> Dict[str, Any]:
        """
        Compute performance metrics for visualization by querying records, prompts, and rounds.
        """
        try:
            # Use try-except pattern instead of checking response.error
            success_records, all_records = self._safe_execute(
                self.client.table("archer_records").select("*"),
                "fetching records"
            )
            
            success_prompts, all_prompts = self._safe_execute(
                self.client.table("archer_prompts").select("*").eq("prompt_type", "generator"),
                "fetching prompts"
            )
            
            success_rounds, all_rounds = self._safe_execute(
                self.client.table("archer_rounds").select("*"),
                "fetching rounds"
            )

            if not success_records or not success_prompts or not success_rounds:
                logger.error("Error fetching performance metrics data")
                return {"rounds": [], "prompts": [], "scores": [], "prompt_survivorship": {}, "moving_avg": []}

            # Convert to empty lists if None
            all_records = all_records or []
            all_prompts = all_prompts or []
            all_rounds = all_rounds or []

            metrics = {
                "rounds": [],
                "prompts": [],
                "scores": [],
                "prompt_survivorship": {}
            }

            # Process generator prompts
            for prompt in all_prompts:
                prompt_id = prompt.get("id", "unknown")
                parent_id = prompt.get("parent_prompt_id", "root")
                avg_score = float(prompt.get("average_score") or 0)
                rounds_survived = int(prompt.get("rounds_survived") or 0)
                is_active = prompt.get("is_active", False)

                metrics["prompts"].append({
                    "id": prompt_id,
                    "parent_id": parent_id,
                    "avg_score": avg_score,
                    "rounds_survived": rounds_survived,
                    "is_active": is_active,
                    "content": prompt.get("content")
                })

                if prompt_id not in metrics["prompt_survivorship"]:
                    metrics["prompt_survivorship"][prompt_id] = {
                        "generations": [rounds_survived],
                        "scores": [avg_score]
                    }
                else:
                    metrics["prompt_survivorship"][prompt_id]["generations"].append(rounds_survived)
                    metrics["prompt_survivorship"][prompt_id]["scores"].append(avg_score)

            # Process records and extract AI scores
            for record in all_records:
                round_id = record.get("round_id", "unknown")
                ai_score = record.get("ai_score")
                if ai_score is not None:
                    round_number = None
                    for round_data in all_rounds:
                        if round_data.get("id") == round_id:
                            round_number = int(round_data.get("round_number", 0))
                            break
                    if round_number is not None:
                        metrics["rounds"].append(round_number)
                        metrics["scores"].append(ai_score)

            if metrics["scores"]:
                window_size = min(5, len(metrics["scores"]))
                metrics["moving_avg"] = np.convolve(metrics["scores"], np.ones(window_size)/window_size, mode='valid').tolist()
            else:
                metrics["moving_avg"] = []

            logger.info(f"Retrieved performance metrics with {len(metrics['prompts'])} prompts")
            return metrics
        except Exception as e:
            logger.error(f"Exception in get_performance_metrics: {str(e)}")
            return {"rounds": [], "prompts": [], "scores": [], "prompt_survivorship": {}, "moving_avg": []}

    def get_prompt_history(self) -> Optional[pd.DataFrame]:
        """
        Retrieve history of generator prompts from the archer_prompts table.
        """
        try:
            success, all_prompts = self._safe_execute(
                self.client.table("archer_prompts").select("*").eq("prompt_type", "generator"),
                "fetching prompt history"
            )
            
            if not success:
                return None
                
            all_prompts = all_prompts or []
            rows = []
            for prompt in all_prompts:
                row = {
                    "prompt_id": prompt.get("id", "unknown"),
                    "parent_prompt_id": prompt.get("parent_prompt_id", "root"),
                    "content": prompt.get("content"),
                    "average_score": float(prompt.get("average_score") or 0),
                    "rounds_survived": int(prompt.get("rounds_survived") or 0),
                    "is_active": prompt.get("is_active", False),
                    "version": int(prompt.get("version") or 1),
                    "created_at": prompt.get("created_at", datetime.now().isoformat())
                }
                rows.append(row)
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values(by=["version", "created_at"])
            logger.info(f"Retrieved prompt history with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Exception in get_prompt_history: {str(e)}")
            return None

    def get_current_best_prompts(self, top_n: int = 4) -> List[str]:
        """
        Retrieve the top N generator prompts based on average score.
        """
        try:
            response = self.client.table("archer_prompts").select("id, average_score")\
                .eq("prompt_type", "generator").execute()
            if response.error:
                logger.error(f"Error fetching best prompts: {response.error}")
                return []
            prompts = response.data or []
            prompts.sort(key=lambda x: float(x.get("average_score") or 0), reverse=True)
            top_prompts = [p.get("id") for p in prompts[:top_n]]
            logger.info(f"Retrieved top {len(top_prompts)} prompts")
            return top_prompts
        except Exception as e:
            logger.error(f"Exception in get_current_best_prompts: {str(e)}")
            return []

    def _get_output(self, output_id: str) -> Optional[Dict]:
        """
        Helper method to fetch an output record by its ID.
        """
        try:
            success, data = self._safe_execute(
                self.client.table("archer_outputs").select("*").eq("id", output_id),
                "fetching output"
            )
            
            if not success:
                return None
                
            data = data or []
            if not data:
                logger.warning(f"No output found with ID: {output_id}")
                return None
            return data[0]
        except Exception as e:
            logger.error(f"Exception in _get_output: {str(e)}")
            return None

    def _get_prompt_text(self, prompt_id: str) -> Optional[str]:
        """
        Helper method to fetch the content of a prompt by its ID.
        """
        try:
            success, data = self._safe_execute(
                self.client.table("archer_prompts").select("content").eq("id", prompt_id),
                "fetching prompt text"
            )
            
            if not success:
                return None
                
            data = data or []
            if not data:
                logger.warning(f"No prompt found with ID: {prompt_id}")
                return None
            return data[0].get("content")
        except Exception as e:
            logger.error(f"Exception in _get_prompt_text: {str(e)}")
            return None

    def _get_latest_evaluation(self, output_id: str) -> Optional[Dict]:
        """
        Helper method to fetch the latest evaluation for a given output.
        """
        try:
            success, data = self._safe_execute(
                self.client.table("archer_evaluations").select("*")
                    .eq("output_id", output_id).order("timestamp", desc=True),
                "fetching latest evaluation"
            )
            
            if not success:
                return None
                
            data = data or []
            if not data:
                logger.debug(f"No evaluations found for output ID: {output_id}")
                return None
            latest = data[0]
            evaluation = {
                "score": latest.get("score"),
                "feedback": latest.get("feedback"),
                "improved_output": latest.get("improved_output")
            }
            return evaluation
        except Exception as e:
            logger.error(f"Exception in _get_latest_evaluation: {str(e)}")
            return None

    def get_validated_evaluations(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Retrieve evaluations that have been validated by humans.
        
        Args:
            limit: Maximum number of evaluations to retrieve
            
        Returns:
            DataFrame containing validated evaluations with all required fields
        """
        try:
            success, records = self._safe_execute(
                self.client.table("archer_evaluations").select("*")
                    .eq("is_human", True).order("timestamp", desc=True).limit(limit),
                "fetching validated evaluations"
            )
            
            if not success:
                return pd.DataFrame()
                
            records = records or []
            rows = []
            for record in records:
                row = {
                    "output_id": record.get("output_id", ""),
                    "prompt_id": record.get("prompt_id", ""),
                    "input": record.get("input", ""),
                    "generated_content": record.get("generated_content", ""),
                    "score": record.get("score", 0),
                    "feedback": record.get("feedback", ""),
                    "improved_output": record.get("improved_output", ""),
                    "timestamp": record.get("timestamp", ""),
                    "evaluator_id": record.get("evaluator_id", "")
                }
                
                # Skip if missing required data
                if not row["prompt_id"]:
                    logger.warning(f"Skipping evaluation with missing prompt_id for output: {row['output_id']}")
                    continue
                
                rows.append(row)
                
            df = pd.DataFrame(rows)
            logger.info(f"Retrieved {len(df)} validated evaluations")
            return df
        except Exception as e:
            logger.error(f"Exception in get_validated_evaluations: {str(e)}")
            return pd.DataFrame()

    def store_record(self, input_data: str, content: str, generator_prompt_id: str, evaluator_prompt_id: str,
                     prompt_generation: int, round_id: str) -> Optional[str]:
        """
        Store a new record in the archer_records table.
        
        Args:
            input_data: The input data used for generation
            content: The generated content
            generator_prompt_id: ID of the generator prompt used
            evaluator_prompt_id: ID of the evaluator prompt used
            prompt_generation: Generation number of the prompt
            round_id: ID of the round
            
        Returns:
            The record ID if successful, None otherwise
        """
        try:
            record_id = str(uuid.uuid4())
            data = {
                "id": record_id,
                "input": input_data,
                "generated_content": content,
                "generator_prompt_id": generator_prompt_id,
                "evaluator_prompt_id": evaluator_prompt_id,
                "prompt_generation": prompt_generation,
                "round_id": round_id,
                "validated_status": False,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            success, _ = self._safe_execute(
                self.client.table("archer_records").insert(data),
                "storing record"
            )
            
            if not success:
                return None
                
            logger.info(f"Stored record with ID: {record_id}")
            return record_id
        except Exception as e:
            logger.error(f"Exception in store_record: {str(e)}")
            return None

    def update_record_evaluation(self, record_id: str, ai_score: float, ai_feedback: str, ai_improved_output: str) -> bool:
        """
        Update a record with AI evaluation data.
        """
        try:
            # Ensure score is an integer
            try:
                score_int = int(float(ai_score))
            except (ValueError, TypeError):
                logger.error(f"Invalid score value: {ai_score}. Must be convertible to integer.")
                return False
                
            data = {
                "ai_score": score_int,
                "ai_feedback": ai_feedback,
                "ai_improved_output": ai_improved_output,
                "updated_at": datetime.now().isoformat()
            }
            
            success, _ = self._safe_execute(
                self.client.table("archer_records").update(data).eq("id", record_id),
                "updating record evaluation"
            )
            
            if not success:
                return False
                
            logger.info(f"Updated record with AI evaluation: {record_id}")
            return True
        except Exception as e:
            logger.error(f"Exception in update_record_evaluation: {str(e)}")
            return False

    def update_record_human_feedback(self, record_id: str, human_score: float, human_feedback: str, human_improved_output: str) -> bool:
        """
        Update a record with human feedback data.
        """
        try:
            # Ensure score is an integer
            try:
                score_int = int(float(human_score))
            except (ValueError, TypeError):
                logger.error(f"Invalid score value: {human_score}. Must be convertible to integer.")
                return False
                
            data = {
                "human_score": score_int,
                "human_feedback": human_feedback,
                "human_improved_output": human_improved_output,
                "validated_status": True,
                "updated_at": datetime.now().isoformat()
            }
            
            success, _ = self._safe_execute(
                self.client.table("archer_records").update(data).eq("id", record_id),
                "updating record with human feedback"
            )
            
            if not success:
                return False
                
            logger.info(f"Updated record with human feedback: {record_id}")
            return True
        except Exception as e:
            logger.error(f"Exception in update_record_human_feedback: {str(e)}")
            return False

    def create_round(self, round_number: int) -> Optional[str]:
        """
        Create a new round in the archer_rounds table.
        """
        try:
            round_id = str(uuid.uuid4())
            data = {
                "id": round_id,
                "round_number": round_number,
                "status": "in_progress",
                "metrics": {},
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            success, _ = self._safe_execute(
                self.client.table("archer_rounds").insert(data),
                "creating round"
            )
            
            if not success:
                return None
                
            logger.info(f"Created round with ID: {round_id}")
            return round_id
        except Exception as e:
            logger.error(f"Exception in create_round: {str(e)}")
            return None

    def update_round(self, round_id: str, status: str = None, metrics: Dict[str, Any] = None) -> bool:
        """
        Update a round's status and metrics in the archer_rounds table.
        """
        try:
            data = {}
            if status:
                data["status"] = status
            if status == "completed":
                data["end_time"] = datetime.now().isoformat()
            if metrics:
                data["metrics"] = json.dumps(metrics)
            if data:
                data["updated_at"] = datetime.now().isoformat()
                
            success, _ = self._safe_execute(
                self.client.table("archer_rounds").update(data).eq("id", round_id),
                "updating round"
            )
            
            if not success:
                return False
                
            logger.info(f"Updated round with ID: {round_id}")
            return True
        except Exception as e:
            logger.error(f"Exception in update_round: {str(e)}")
            return False

    def store_prompt_lineage(self, parent_prompt_id: str, child_prompt_id: str, round_id: str, change_reason: str) -> Optional[str]:
        """
        Store prompt lineage information in the archer_prompt_lineage table.
        """
        try:
            lineage_id = str(uuid.uuid4())
            data = {
                "id": lineage_id,
                "parent_prompt_id": parent_prompt_id,
                "child_prompt_id": child_prompt_id,
                "round_id": round_id,
                "change_reason": change_reason,
                "created_at": datetime.now().isoformat()
            }
            
            success, _ = self._safe_execute(
                self.client.table("archer_prompt_lineage").insert(data),
                "storing prompt lineage"
            )
            
            if not success:
                return None
                
            logger.info(f"Stored prompt lineage with ID: {lineage_id}")
            return lineage_id
        except Exception as e:
            logger.error(f"Exception in store_prompt_lineage: {str(e)}")
            return None

    def _get_record(self, record_id: str) -> Optional[Dict]:
        """
        Helper method to fetch a record by its ID from archer_records.
        """
        try:
            response = self.client.table("archer_records").select("*").eq("id", record_id).execute()
            if response.error:
                logger.error(f"Error fetching record: {response.error}")
                return None
            data = response.data or []
            if not data:
                logger.warning(f"No record found with ID: {record_id}")
                return None
            return data[0]
        except Exception as e:
            logger.error(f"Exception in _get_record: {str(e)}")
            return None

    def _get_generator_prompt(self, prompt_id: str) -> Optional[Dict]:
        """
        Helper method to fetch a generator prompt by its ID.
        """
        try:
            response = self.client.table("archer_prompts").select("*")\
                .eq("id", prompt_id).eq("prompt_type", "generator").execute()
            if response.error:
                logger.error(f"Error fetching generator prompt: {response.error}")
                return None
            data = response.data or []
            if not data:
                logger.warning(f"No generator prompt found with ID: {prompt_id}")
                return None
            return data[0]
        except Exception as e:
            logger.error(f"Exception in _get_generator_prompt: {str(e)}")
            return None

    def _get_evaluator_prompt(self, prompt_id: str) -> Optional[Dict]:
        """
        Helper method to fetch an evaluator prompt by its ID.
        """
        try:
            response = self.client.table("archer_prompts").select("*")\
                .eq("id", prompt_id).eq("prompt_type", "evaluator").execute()
            if response.error:
                logger.error(f"Error fetching evaluator prompt: {response.error}")
                return None
            data = response.data or []
            if not data:
                logger.warning(f"No evaluator prompt found with ID: {prompt_id}")
                return None
            return data[0]
        except Exception as e:
            logger.error(f"Exception in _get_evaluator_prompt: {str(e)}")
            return None

    def _get_round(self, round_id: str) -> Optional[Dict]:
        """
        Helper method to fetch a round by its ID from archer_rounds.
        """
        try:
            response = self.client.table("archer_rounds").select("*").eq("id", round_id).execute()
            if response.error:
                logger.error(f"Error fetching round: {response.error}")
                return None
            data = response.data or []
            if not data:
                logger.warning(f"No round found with ID: {round_id}")
                return None
            return data[0]
        except Exception as e:
            logger.error(f"Exception in _get_round: {str(e)}")
            return None

    def get_active_evaluator_prompts(self) -> List[Dict[str, Any]]:
        """
        Retrieve all active evaluator prompts from archer_prompts.
        """
        try:
            response = self.client.table("archer_prompts").select("*")\
                .eq("prompt_type", "evaluator").eq("is_active", True).execute()
            if response.error:
                logger.error(f"Error fetching active evaluator prompts: {response.error}")
                return []
            active_prompts = response.data or []
            prompt_data = []
            for prompt in active_prompts:
                prompt_data.append({
                    "id": prompt.get("id", "unknown"),
                    "content": prompt.get("content"),
                    "version": int(prompt.get("version") or 1),
                    "created_at": prompt.get("created_at", datetime.now().isoformat())
                })
            logger.info(f"Retrieved {len(prompt_data)} active evaluator prompts")
            return prompt_data
        except Exception as e:
            logger.error(f"Exception in get_active_evaluator_prompts: {str(e)}")
            return []

    def get_round_metrics(self, round_id: str) -> Dict[str, Any]:
        """
        Retrieve metrics for a specific round.
        """
        try:
            response = self.client.table("archer_rounds").select("*").eq("id", round_id).execute()
            if response.error:
                logger.error(f"Error fetching round metrics: {response.error}")
                return {}
            data = response.data or []
            if not data:
                logger.warning(f"No round found with ID: {round_id}")
                return {}
            round_data = data[0]
            metrics_json = round_data.get("metrics", "{}")
            try:
                metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else metrics_json
            except json.JSONDecodeError:
                logger.error(f"Error decoding metrics JSON for round {round_id}")
                metrics = {}
            logger.info(f"Retrieved metrics for round ID: {round_id}")
            return metrics
        except Exception as e:
            logger.error(f"Exception in get_round_metrics: {str(e)}")
            return {}

    def get_active_generator_prompts(self, top_n: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve the top N active generator prompts from archer_prompts.
        """
        try:
            response = self.client.table("archer_prompts").select("*")\
                .eq("prompt_type", "generator").eq("is_active", True).execute()
            if response.error:
                logger.error(f"Error fetching active generator prompts: {response.error}")
                return []
            active_prompts = response.data or []
            prompt_data = []
            for prompt in active_prompts:
                prompt_data.append({
                    "id": prompt.get("id", "unknown"),
                    "content": prompt.get("content"),
                    "average_score": float(prompt.get("average_score") or 0),
                    "rounds_survived": int(prompt.get("rounds_survived") or 0),
                    "version": int(prompt.get("version") or 1),
                    "parent_prompt_id": prompt.get("parent_prompt_id", "root")
                })
            prompt_data.sort(key=lambda x: x["average_score"], reverse=True)
            top_prompts = prompt_data[:top_n]
            logger.info(f"Retrieved top {len(top_prompts)} active generator prompts")
            return top_prompts
        except Exception as e:
            logger.error(f"Exception in get_active_generator_prompts: {str(e)}")
            return []

    def get_prompts_from_records(self, prompt_type: str = "generator", generations: List[int] = None, top_n: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve prompts directly from records.
        Filters by prompt type and optionally by specific generations.
        """
        try:
            query = self.client.table("archer_records").select("*")
            if generations:
                # Filter by the prompt_generation field (assuming stored as numbers or strings)
                query = query.in_("prompt_generation", [str(g) for g in generations])
            response = query.execute()
            if response.error:
                logger.error(f"Error fetching records for prompts: {response.error}")
                return []
            results = response.data or []
            prompts_seen = set()
            prompts = []
            for record in results:
                prompt_field = "generator_prompt_id" if prompt_type == "generator" else "evaluator_prompt_id"
                prompt_id = record.get(prompt_field, "")
                if not prompt_id or prompt_id in prompts_seen:
                    continue
                # Fetch prompt details
                prompt_response = self.client.table("archer_prompts").select("*").eq("id", prompt_id).execute()
                if prompt_response.error or not prompt_response.data:
                    continue
                prompt_data = prompt_response.data[0]
                generation_val = int(record.get("prompt_generation", 0))
                round_id = record.get("round_id", "")
                timestamp = record.get("created_at", "")
                score = float(record.get("ai_score", 0))
                prompts_seen.add(prompt_id)
                prompts.append({
                    'content': prompt_data.get("content", ""),
                    'generation': generation_val,
                    'score': score,
                    'round_id': round_id,
                    'timestamp': timestamp
                })
            prompts.sort(key=lambda x: x['score'], reverse=True)
            return prompts[:top_n]
        except Exception as e:
            logger.error(f"Exception in get_prompts_from_records: {str(e)}")
            return []

    def get_all_generator_prompts(self) -> List[Dict[str, Any]]:
        """
        Retrieve all generator prompts from the database, regardless of their active status.
        This allows for random sampling of prompts during generation.
        
        Returns:
            List[Dict[str, Any]]: A list of all generator prompts with their attributes
        """
        try:
            response = self.client.table("archer_prompts").select("*")\
                .eq("prompt_type", "generator").execute()
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error fetching generator prompts: {response.error}")
                return []
            
            all_prompts = response.data or []
            prompt_data = []
            
            for prompt in all_prompts:
                prompt_data.append({
                    "id": prompt.get("id", "unknown"),
                    "content": prompt.get("content"),
                    "average_score": float(prompt.get("average_score") or 0),
                    "rounds_survived": int(prompt.get("rounds_survived") or 0),
                    "version": int(prompt.get("version") or 1),
                    "parent_prompt_id": prompt.get("parent_prompt_id", "root"),
                    "is_active": prompt.get("is_active", True)
                })
            
            logger.info(f"Retrieved {len(prompt_data)} generator prompts from database")
            return prompt_data
        except Exception as e:
            logger.error(f"Exception in get_all_generator_prompts: {str(e)}")
            return []

    def update_prompt_score(self, prompt_id: str, new_score: float) -> bool:
        """
        Update the average score for a prompt based on a new evaluation score.
        This calculates a running average of all scores for this prompt.
        
        Args:
            prompt_id: The ID of the prompt to update
            new_score: The new score to incorporate into the average
            
        Returns:
            bool: True if the update was successful, False otherwise
        """
        try:
            # First get the current prompt data
            response = self.client.table("archer_prompts").select("*").eq("id", prompt_id).execute()
            if hasattr(response, 'error') and response.error:
                logger.error(f"Error fetching prompt to update score: {response.error}")
                return False
                
            if not response.data or len(response.data) == 0:
                logger.error(f"Prompt with ID {prompt_id} not found")
                return False
                
            prompt_data = response.data[0]
            current_avg = float(prompt_data.get("average_score") or 0)
            usage_count = int(prompt_data.get("usage_count") or 0)
            
            # Calculate new average
            if usage_count == 0:
                new_avg = new_score
            else:
                new_avg = (current_avg * usage_count + new_score) / (usage_count + 1)
                
            # Update the prompt with new average and increment usage count
            update_data = {
                "average_score": new_avg,
                "usage_count": usage_count + 1,
                "last_used_at": datetime.now().isoformat()
            }
            
            update_response = self.client.table("archer_prompts").update(update_data).eq("id", prompt_id).execute()
            if hasattr(update_response, 'error') and update_response.error:
                logger.error(f"Error updating prompt score: {update_response.error}")
                return False
                
            logger.info(f"Updated average score for prompt {prompt_id} to {new_avg:.2f} (usage count: {usage_count + 1})")
            return True
            
        except Exception as e:
            logger.error(f"Exception in update_prompt_score: {str(e)}")
            return False

    def fix_missing_prompt_ids(self, limit: int = 100) -> int:
        """
        Fix evaluations with missing prompt IDs by finding or creating appropriate prompt IDs.
        
        Args:
            limit: Maximum number of evaluations to fix
            
        Returns:
            Number of fixed evaluations
        """
        try:
            logger.info(f"Looking for evaluations with missing prompt IDs (limit: {limit})")
            
            # Fetch evaluations with null prompt_id
            success, records = self._safe_execute(
                self.client.table("archer_evaluations").select("*")
                    .is_("prompt_id", "null").limit(limit),
                "fetching evaluations with null prompt_id"
            )
            
            if not success:
                logger.error("Failed to fetch evaluations with null prompt_id")
                return 0
                
            records = records or []
            logger.info(f"Found {len(records)} evaluations with null prompt_id")
            
            if not records:
                return 0
            
            fixed_count = 0
            
            for evaluation in records:
                eval_id = evaluation.get("id")
                output_id = evaluation.get("output_id")
                
                if not output_id:
                    logger.warning(f"Evaluation {eval_id} has null output_id, skipping")
                    continue
                
                logger.info(f"Fixing evaluation {eval_id} for output {output_id}")
                
                # Get the output
                output = self._get_output(output_id)
                if not output:
                    logger.warning(f"Output {output_id} not found for evaluation {eval_id}")
                    continue
                
                # Get prompt ID from output
                prompt_id = output.get("prompt_id")
                
                if not prompt_id:
                    logger.info(f"Output {output_id} also has null prompt_id, searching for a match")
                    
                    # Try to find a prompt by content match in records
                    try:
                        generated_content = output.get("generated_content", "")
                        if generated_content:
                            records_response = self.client.table("archer_records").select("generator_prompt_id").eq("generated_content", generated_content).execute()
                                
                            if records_response and hasattr(records_response, 'data') and records_response.data:
                                prompt_id = records_response.data[0].get("generator_prompt_id")
                                if prompt_id:
                                    logger.info(f"Found prompt ID {prompt_id} by content matching in records")
                    except Exception as e:
                        logger.error(f"Error finding prompt by content: {str(e)}")
                
                # If still no prompt_id, create a new one
                if not prompt_id:
                    logger.info(f"Creating new prompt for evaluation {eval_id}")
                    prompt_content = f"Retroactively created prompt for evaluation {eval_id}"
                    prompt_id = self.store_generator_prompt(content=prompt_content)
                    
                    if not prompt_id:
                        logger.error(f"Failed to create prompt for evaluation {eval_id}")
                        continue
                    
                    # Also update the output with this prompt_id
                    try:
                        self.client.table("archer_outputs").update({"prompt_id": prompt_id}).eq("id", output_id).execute()
                        logger.info(f"Updated output {output_id} with prompt_id {prompt_id}")
                    except Exception as e:
                        logger.error(f"Error updating output with new prompt_id: {str(e)}")
                
                # Update the evaluation with the prompt_id
                try:
                    self.client.table("archer_evaluations").update({"prompt_id": prompt_id}).eq("id", eval_id).execute()
                    logger.info(f"Updated evaluation {eval_id} with prompt_id {prompt_id}")
                    fixed_count += 1
                except Exception as e:
                    logger.error(f"Error updating evaluation with prompt_id: {str(e)}")
            
            logger.info(f"Fixed {fixed_count} evaluations with missing prompt IDs")
            return fixed_count
            
        except Exception as e:
            logger.error(f"Exception in fix_missing_prompt_ids: {str(e)}")
            return 0