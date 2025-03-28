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

    def store_generated_content(self, input_data: str, content: str, prompt_id: str, round_num: int) -> Optional[str]:
        """
        Store generated content in the archer_outputs table.
        """
        try:
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
            if response.error:
                logger.error(f"Error storing generated content: {response.error}")
                return None
            logger.info(f"Stored generated content with ID: {output_id}")
            return output_id
        except Exception as e:
            logger.error(f"Exception in store_generated_content: {str(e)}")
            return None

    def store_evaluation(self, output_id: str, score: int, feedback: str, improved_output: str, is_human: bool = False) -> bool:
        """
        Store an evaluation for a given output in the archer_evaluations table.
        """
        try:
            # Retrieve the original output for reference
            output = self._get_output(output_id)
            if not output:
                logger.error(f"Output with ID {output_id} not found")
                return False

            evaluation_id = str(uuid.uuid4())
            data = {
                "id": evaluation_id,
                "input": output.get("input_data", ""),
                "generated_content": output.get("generated_content", ""),
                "evaluation_content": "",  # Can be expanded if needed
                "score": score,
                "feedback": feedback,
                "improved_output": improved_output,
                "output_id": output_id,
                "prompt_id": output.get("prompt_id"),
                "evaluator_id": "human" if is_human else "ai_evaluator",
                "is_human": is_human,
                "timestamp": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_evaluations").insert(data).execute()
            if response.error:
                logger.error(f"Error storing evaluation: {response.error}")
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
            if response.error:
                logger.error(f"Error storing prompt: {response.error}")
                return None
            logger.info(f"Stored {prompt_type} prompt with ID: {prompt_id}")
            return prompt_id
        except Exception as e:
            logger.error(f"Exception in store_prompt: {str(e)}")
            return None

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
            if response.error:
                logger.error(f"Error updating prompt performance: {response.error}")
                return False
            logger.info(f"Updated performance for prompt ID: {prompt_id}")
            return True
        except Exception as e:
            logger.error(f"Exception in update_prompt_performance: {str(e)}")
            return False

    def get_current_data_for_annotation(self, round_id: str, limit: int = 20) -> Optional[pd.DataFrame]:
        """
        Retrieve current records for human annotation based on a round ID.
        """
        try:
            response = self.client.table("archer_records").select("*").eq("round_id", round_id).execute()
            if response.error:
                logger.error(f"Error fetching records for annotation: {response.error}")
                return pd.DataFrame()
            records = response.data or []
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
            records_response = self.client.table("archer_records").select("*").execute()
            prompts_response = self.client.table("archer_prompts").select("*").eq("prompt_type", "generator").execute()
            rounds_response = self.client.table("archer_rounds").select("*").execute()

            if records_response.error or prompts_response.error or rounds_response.error:
                logger.error("Error fetching performance metrics data")
                return {"rounds": [], "prompts": [], "scores": [], "prompt_survivorship": {}, "moving_avg": []}

            all_records = records_response.data or []
            all_prompts = prompts_response.data or []
            all_rounds = rounds_response.data or []

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
            response = self.client.table("archer_prompts").select("*").eq("prompt_type", "generator").execute()
            if response.error:
                logger.error(f"Error fetching prompt history: {response.error}")
                return None
            all_prompts = response.data or []
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
            response = self.client.table("archer_outputs").select("*").eq("id", output_id).execute()
            if response.error:
                logger.error(f"Error fetching output: {response.error}")
                return None
            data = response.data or []
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
            response = self.client.table("archer_prompts").select("content").eq("id", prompt_id).execute()
            if response.error:
                logger.error(f"Error fetching prompt text: {response.error}")
                return None
            data = response.data or []
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
            response = self.client.table("archer_evaluations").select("*")\
                .eq("output_id", output_id).order("timestamp", desc=True).execute()
            if response.error:
                logger.error(f"Error fetching latest evaluation: {response.error}")
                return None
            data = response.data or []
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
        """
        try:
            response = self.client.table("archer_evaluations").select("*")\
                .eq("is_human", True).order("timestamp", desc=True).limit(limit).execute()
            if response.error:
                logger.error(f"Error fetching validated evaluations: {response.error}")
                return pd.DataFrame()
            records = response.data or []
            rows = []
            for record in records:
                row = {
                    "output_id": record.get("output_id"),
                    "prompt_id": record.get("prompt_id"),
                    "input": record.get("input"),
                    "generated_content": record.get("generated_content"),
                    "score": record.get("score"),
                    "feedback": record.get("feedback"),
                    "improved_output": record.get("improved_output"),
                    "timestamp": record.get("timestamp")
                }
                rows.append(row)
            df = pd.DataFrame(rows)
            logger.info(f"Retrieved {len(df)} validated evaluations")
            return df
        except Exception as e:
            logger.error(f"Exception in get_validated_evaluations: {str(e)}")
            return None

    def store_record(self, input_data: str, content: str, generator_prompt: str, evaluator_prompt: str,
                     prompt_generation: int, round_id: str) -> Optional[str]:
        """
        Store a new record in the archer_records table.
        """
        try:
            record_id = str(uuid.uuid4())
            data = {
                "id": record_id,
                "input": input_data,
                "generated_content": content,
                "generator_prompt_id": generator_prompt,
                "evaluator_prompt_id": evaluator_prompt,
                "prompt_generation": prompt_generation,
                "round_id": round_id,
                "validated_status": False,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_records").insert(data).execute()
            if response.error:
                logger.error(f"Error storing record: {response.error}")
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
            data = {
                "ai_score": ai_score,
                "ai_feedback": ai_feedback,
                "ai_improved_output": ai_improved_output,
                "updated_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_records").update(data).eq("id", record_id).execute()
            if response.error:
                logger.error(f"Error updating record evaluation: {response.error}")
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
            data = {
                "human_score": human_score,
                "human_feedback": human_feedback,
                "human_improved_output": human_improved_output,
                "validated_status": True,
                "updated_at": datetime.now().isoformat()
            }
            response = self.client.table("archer_records").update(data).eq("id", record_id).execute()
            if response.error:
                logger.error(f"Error updating record with human feedback: {response.error}")
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
            response = self.client.table("archer_rounds").insert(data).execute()
            if response.error:
                logger.error(f"Error creating round: {response.error}")
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
            response = self.client.table("archer_rounds").update(data).eq("id", round_id).execute()
            if response.error:
                logger.error(f"Error updating round: {response.error}")
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
            response = self.client.table("archer_prompt_lineage").insert(data).execute()
            if response.error:
                logger.error(f"Error storing prompt lineage: {response.error}")
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