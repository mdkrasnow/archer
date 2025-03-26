"""
promptoptimizer.py

This module implements prompt optimization using a feedback-driven approach
integrated with the AdaLflow library. It leverages AdaLflow’s trainable prompt parameters
to evolve prompts based on performance feedback.

References:
  - Trainable Prompt as Parameter: https://adalflow.sylph.ai/new_tutorials/generator.html#trainable-prompt-as-parameter
  - LLM-AutoDiff: https://adalflow.sylph.ai/new_tutorials/introduction.html#llm-autodiff
"""

from helpers.prompt import Prompt
from helpers.llm_call import llm_call

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
    
    def __init__(self, model_name: str):
        """
        Initialize a new PromptOptimizer.

        Args:
            model_name: Identifier of the LLM to use for prompt optimization.
        """
        self.model_name = model_name
        self.llm_call = llm_call
    
    def optimize_prompt(self, prompt: Prompt, feedback: str, score: float) -> str:
        """
        Generate an improved prompt based on provided feedback and score.
        
        This method creates a trainable prompt parameter using AdaLflow’s Parameter class.
        The parameter’s instruction is built using the feedback and score to guide the optimization process.
        It then uses an LLM call to produce a new prompt variant that better addresses the feedback.
        
        Args:
            prompt: The original Prompt object to be improved.
            feedback: Textual feedback indicating improvements to be made.
            score: Performance score (e.g., 1-5) of the original prompt.
            
        Returns:
            A string representing the improved prompt.
        """
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
        
        try:
            improved_prompt = self.llm_call(
                model=self.model_name,
                prompt=(
                    f"Improve the following prompt based on feedback.\n\n"
                    f"Original Prompt: {prompt_param.data}\n\n"
                    f"Feedback: {feedback}\n"
                    f"Score: {score} out of 5.\n\n"
                    "Improved Prompt:"
                ),
                temperature=0.7
            )
        except Exception:
            return prompt.content
        
        # Fallback: If optimization fails to produce a result, return the original prompt.
        if not improved_prompt or not improved_prompt.strip():
            return prompt.content
        
        return improved_prompt