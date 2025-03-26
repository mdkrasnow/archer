"""
This module defines the GenerativeModel class for generating content using LLMs.
"""

from archer.helpers.llm_call import llm_call as default_llm_call

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
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.active_prompts = []
        self.generation_func = generation_func
        self.llm_call = llm_call

    def set_prompts(self, prompts):
        """
        Set the prompts to use for generation.
        
        Args:
            prompts (list): List of Prompt objects.
        """
        self.active_prompts = prompts
    
    def _call_llm(self, prompt, input_data):
        """
        Make a call to the language model.
        
        Args:
            prompt (str): The prompt content to use.
            input_data (str): The input data to process.
            
        Returns:
            str: The generated content or error message.
        """
        try:
            # Use the custom llm_call if provided, otherwise use the imported one
            llm_call_func = self.llm_call if self.llm_call else default_llm_call
            
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
                return response["choices"][0]["message"]["content"]
            else:
                return "Error: No response generated"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate(self, input_data):
        """
        Generate content for the given input data using the stored prompts.
        
        Args:
            input_data: The input data to feed into the generation process.
            
        Returns:
            list: A list of tuples (generated_content, prompt).
        """
        results = []
        
        # If no prompts are set, return empty results
        if not self.active_prompts:
            return results
            
        for prompt in self.active_prompts:
            if self.generation_func:
                # Use custom generation function if provided
                generated_content = self.generation_func(prompt.content, input_data)
            else:
                # Use default LLM call
                generated_content = self._call_llm(prompt.content, input_data)
                
            results.append((generated_content, prompt))
        
        return results 