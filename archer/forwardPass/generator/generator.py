"""
This module provides the GenerativeModel class for generating content with language models.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from archer.helpers.llm_call import llm_call as default_llm_call
from ...helpers.logging_utils import get_logger, log_entry_exit, log_call_args

# Setup logger for this module
logger = get_logger(__name__)

class GenerativeModel:
    """
    A class for generating content using a specific language model.
    
    Attributes:
        model_name (str): The name of the LLM to use.
        temperature (float): The temperature parameter for generation.
        top_p (float): The top_p parameter for generation.
        active_prompts (list): A list of Prompt objects to use for generation.
    """
    
    def __init__(self, model_name, temperature=0.7, top_p=0.9, generation_func=None, llm_call=None):
        """
        Initialize a new GenerativeModel.

        Args:
            model_name (str): Name/identifier of the LLM (e.g., "gpt-4").
            temperature (float, optional): Temperature setting for generation.
            top_p (float, optional): Top-p setting for generation.
            generation_func (callable, optional): Custom generation function.
            llm_call (callable, optional): Custom LLM call function.
        """
        logger.info(f"Initializing GenerativeModel with model: {model_name}")
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.active_prompts = []
        self.generation_func = generation_func
        self.llm_call = llm_call
        logger.debug(f"GenerativeModel initialized with temperature={temperature}, top_p={top_p}")

    @log_entry_exit(logger)
    def set_prompts(self, prompts):
        """
        Set the prompts to use for generation.
        
        Args:
            prompts (list): List of Prompt objects.
        """
        logger.info(f"Setting {len(prompts)} prompts for generation")
        self.active_prompts = prompts
    
    @log_entry_exit(logger)
    def _call_llm(self, prompt, input_data):
        """
        Make a call to the language model.
        
        Args:
            prompt (str): The prompt content to use.
            input_data (str): The input data to process.
            
        Returns:
            str: The generated content or error message.
        """
        logger.debug(f"Calling LLM {self.model_name} with temperature {self.temperature}")
        logger.debug(f"Prompt: {prompt[:100]}...")
        
        try:
            # Use the custom llm_call if provided, otherwise use the imported one
            llm_call_func = self.llm_call if self.llm_call else default_llm_call
            
            logger.debug("Making API call to LLM")
            response = llm_call_func(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": input_data}
                ],
                model=self.model_name,
                temperature=self.temperature
            )
            
            # Extract content from the response
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                logger.debug(f"LLM response received, length: {len(content)} chars")
                return content
            else:
                logger.warning("No valid response received from LLM")
                return "Error: No response generated"
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error: {str(e)}"
    
    @log_entry_exit(logger)
    def generate(self, input_data):
        """
        Generate content for the given input data using the stored prompts.
        
        Args:
            input_data: The input data to feed into the generation process.
            
        Returns:
            list: A list of tuples (generated_content, prompt).
        """
        prompt_count = len(self.active_prompts)
        logger.info(f"Generating content using {prompt_count} prompts")
        
        results = []
        
        # If no prompts are set, return empty results
        if not self.active_prompts:
            logger.warning("No active prompts set, returning empty results")
            return results
            
        for i, prompt in enumerate(self.active_prompts):
            logger.info(f"Generating with prompt {i+1}/{prompt_count}: {prompt.content[:50]}...")
            
            if self.generation_func:
                # Use custom generation function if provided
                logger.debug("Using custom generation function")
                generated_content = self.generation_func(prompt.content, input_data)
            else:
                # Use default LLM call
                logger.debug("Using default LLM call")
                generated_content = self._call_llm(prompt.content, input_data)
                
            content_length = len(generated_content)
            logger.info(f"Generated content with prompt {i+1}, length: {content_length} chars")
            
            results.append((generated_content, prompt))
        
        logger.info(f"Generation complete, produced {len(results)} outputs")
        return results 