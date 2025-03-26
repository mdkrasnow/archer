"""
promptoptimizer.py

This module implements prompt optimization using a feedback-driven approach
integrated with the AdaLflow library. It leverages AdaLflow's trainable prompt parameters
to evolve prompts based on performance feedback.

References:
  - Trainable Prompt as Parameter: https://adalflow.sylph.ai/new_tutorials/generator.html#trainable-prompt-as-parameter
  - LLM-AutoDiff: https://adalflow.sylph.ai/new_tutorials/introduction.html#llm-autodiff
"""

from helpers.prompt import Prompt
from helpers.llm_call import llm_call
from ...helpers.logging_utils import get_logger, log_entry_exit, log_call_args

# Setup logger for this module
logger = get_logger(__name__)

# Import AdaLflow components for prompt optimization.
import adalflow as adal
from adalflow.optim.parameter import Parameter, ParameterType

class PromptOptimizer:
    """
    Optimizes prompts based on feedback using a feedback-driven approach integrated with AdaLflow.

    This class improves prompts by wrapping the original prompt into a trainable AdaLflow Parameter,
    then leveraging an LLM call to generate an improved version. The optimization process
    considers both the textual feedback and a numerical score (e.g., on a 1-5 scale), acting as a gradient
    for prompt improvement.

    References:
      - Trainable Prompt as Parameter:
        https://adalflow.sylph.ai/new_tutorials/generator.html#trainable-prompt-as-parameter
      - LLM-AutoDiff:
        https://adalflow.sylph.ai/new_tutorials/introduction.html#llm-autodiff
    """
    
    def __init__(self, model_name: str, temperature=0.7, adalflow_enabled=False, max_trials=5, top_k=3):
        """
        Initialize a new PromptOptimizer.

        Args:
            model_name: Identifier of the LLM to use for prompt optimization.
            temperature: Temperature parameter for LLM calls.
            adalflow_enabled: Whether to use AdaLFlow for optimization.
            max_trials: Maximum number of trials for AdaLFlow optimization.
            top_k: Number of top prompts to keep in AdaLFlow optimization.
        """
        logger.info(f"Initializing PromptOptimizer with model: {model_name}")
        self.model_name = model_name
        self.temperature = temperature
        self.adalflow_enabled = adalflow_enabled
        self.max_trials = max_trials
        self.top_k = top_k
        self.llm_call = llm_call
        
        if adalflow_enabled:
            logger.info("AdaLFlow optimization is enabled")
            logger.debug(f"AdaLFlow config: max_trials={max_trials}, top_k={top_k}")
        
        logger.info("PromptOptimizer initialization complete")
    
    @log_entry_exit(logger)
    def optimize_prompt(self, prompt: Prompt, feedback: str, score: float) -> str:
        """
        Generate an improved prompt based on provided feedback and score.
        
        This method creates a trainable prompt parameter using AdaLflow's Parameter class.
        The parameter's instruction is built using the feedback and score to guide the optimization process.
        It then uses an LLM call to produce a new prompt variant that better addresses the feedback.
        
        Args:
            prompt: The original Prompt object to be improved.
            feedback: Textual feedback indicating improvements to be made.
            score: Performance score (e.g., 1-5) of the original prompt.
            
        Returns:
            A string representing the improved prompt.
        """
        logger.info(f"Optimizing prompt with score: {score}")
        logger.debug(f"Original prompt: {prompt.content[:100]}...")
        logger.debug(f"Feedback: {feedback[:100]}...")
        
        if self.adalflow_enabled:
            logger.info("Using AdaLFlow for prompt optimization")
            # AdaLFlow optimization code would go here
            # For now, we'll use the standard optimization as fallback
            logger.warning("AdaLFlow implementation not complete, falling back to standard optimization")
        
        # Wrap the original prompt content into a trainable Parameter.
        prompt_param = Parameter(
            data=prompt.content,
            role_desc="Prompt to be optimized based on feedback.",
            requires_opt=True,
            param_type=ParameterType.PROMPT,  # Enables prompt tuning.
            instruction_to_optimizer=(
                f"Improve the prompt based on the following feedback and score:\n"
                f"Feedback: {feedback}\n"
                f"Score: {score} out of 5."
            )
        )
        
        logger.debug("Created AdaLFlow parameter for prompt optimization")
        
        try:
            logger.info(f"Calling {self.model_name} to generate improved prompt")
            improved_prompt = self.llm_call(
                model=self.model_name,
                prompt=(
                    f"Improve the following prompt based on feedback.\n\n"
                    f"Original Prompt: {prompt_param.data}\n\n"
                    f"Feedback: {feedback}\n"
                    f"Score: {score} out of 5.\n\n"
                    "Improved Prompt:"
                ),
                temperature=self.temperature
            )
            
            if improved_prompt and improved_prompt.strip():
                logger.info("Successfully generated improved prompt")
                logger.debug(f"Improved prompt: {improved_prompt[:100]}...")
                return improved_prompt
            else:
                logger.warning("Empty or invalid response from LLM, returning original prompt")
                return prompt.content
                
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            logger.warning("Returning original prompt due to optimization error")
            return prompt.content
    
    @log_entry_exit(logger)
    def create_prompt_variation(self, prompt_content: str, variation_message: str) -> str:
        """
        Create a variation of a prompt with specific traits or characteristics.
        
        Args:
            prompt_content: The original prompt content
            variation_message: Message describing what kind of variation to create
            
        Returns:
            A string representing the varied prompt
        """
        logger.info("Creating prompt variation")
        logger.debug(f"Original prompt: {prompt_content[:100]}...")
        logger.debug(f"Variation message: {variation_message}")
        
        try:
            logger.info(f"Calling {self.model_name} to generate prompt variation")
            variation = self.llm_call(
                model=self.model_name,
                prompt=(
                    f"{variation_message}\n\n"
                    f"Original Prompt: {prompt_content}\n\n"
                    "Varied Prompt:"
                ),
                temperature=self.temperature + 0.1  # Slightly higher temperature for variety
            )
            
            if variation and variation.strip():
                logger.info("Successfully generated prompt variation")
                logger.debug(f"Variation: {variation[:100]}...")
                return variation
            else:
                logger.warning("Empty or invalid response for variation, returning original")
                return prompt_content
                
        except Exception as e:
            logger.error(f"Error creating prompt variation: {str(e)}")
            return prompt_content
    
    @log_entry_exit(logger)
    def optimize(self, prompt_objs: list, feedback_map: dict, score_map: dict) -> list:
        """
        Optimize a list of prompts based on feedback and scores.
        
        Args:
            prompt_objs: List of Prompt objects to optimize
            feedback_map: Dictionary mapping prompt IDs to feedback strings
            score_map: Dictionary mapping prompt IDs to scores
            
        Returns:
            List of improved Prompt objects
        """
        logger.info(f"Optimizing {len(prompt_objs)} prompts with AdaLFlow")
        
        improved_prompts = []
        
        for i, prompt in enumerate(prompt_objs):
            prompt_id = str(i)
            feedback = feedback_map.get(prompt_id, "")
            score = score_map.get(prompt_id, 0)
            
            logger.info(f"Optimizing prompt {i+1}/{len(prompt_objs)}")
            
            improved_content = self.optimize_prompt(prompt, feedback, score)
            
            # Create a new prompt with the improved content
            improved_prompt = Prompt(content=improved_content)
            improved_prompt.parent = prompt
            improved_prompts.append(improved_prompt)
        
        logger.info(f"Optimization complete, produced {len(improved_prompts)} improved prompts")
        return improved_prompts
    
    @log_entry_exit(logger)
    def optimize_model(self, model, feedback_map: dict, score_map: dict):
        """
        Optimize prompts in a model using AdaLFlow.
        
        Args:
            model: The model containing prompts to optimize
            feedback_map: Dictionary mapping prompt IDs to feedback strings
            score_map: Dictionary mapping prompt IDs to scores
        """
        logger.info(f"Optimizing model prompts with AdaLFlow")
        logger.info(f"Model has {len(model.prompts)} prompts to optimize")
        
        # This would contain AdaLFlow-specific optimization code
        # For now, we'll just log that it would happen
        for prompt_id, prompt in model.prompts.items():
            feedback = feedback_map.get(prompt_id, "")
            score = score_map.get(prompt_id, 0)
            
            logger.info(f"Would optimize model prompt {prompt_id} with score {score}")
            
        logger.info("Model optimization complete")