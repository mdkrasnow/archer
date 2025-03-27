"""
This module defines the AIExpert class for evaluating generated content.
"""

from typing import List, Dict, Any
from archer.helpers.llm_call import llm_call
import os
class AIExpert:
    """
    A class for evaluating generated content against a rubric.
    
    Attributes:
        model_name (str): The name of the LLM to use for evaluation.
        knowledge_base (list): A list of document contents for RAG.
        rubric (dict): Evaluation criteria.
        current_prompt (str): The currently active evaluator prompt.
    """
    
    def __init__(self, model_name, knowledge_base, rubric):
        """
        Initialize a new AIExpert.
        
        Args:
            model_name (str): Name/identifier of the LLM.
            knowledge_base (list): List of document contents for RAG.
            rubric (dict): Evaluation criteria as a dictionary.
        """
        self.model_name = model_name
        self.knowledge_base = knowledge_base
        self.rubric = rubric
        self.llm_call = llm_call
        self.current_prompt = ""
        self._set_default_prompt()
    
    def _set_default_prompt(self):
        """Set the default evaluator prompt with placeholders for input and content."""
        self.current_prompt = f"""
        You are an expert evaluator. Please evaluate the following generated content 
        against the provided rubric and input data.
        
        Input data: {{input_placeholder}}
        
        Generated content: {{content_placeholder}}
        
        Rubric: {self.rubric}
        
        Please provide:
        1. A score from 1-5
        2. Detailed feedback on how to improve
        3. An example of improved output
        4. A brief summary of your evaluation
        """
    
    def set_prompt(self, prompt_content):
        """
        Set the evaluator prompt.
        
        Args:
            prompt_content (str): New prompt content with placeholders.
        """
        self.current_prompt = prompt_content
    
    def get_current_prompt(self):
        """
        Get the current evaluator prompt.
        
        Returns:
            str: The current evaluator prompt.
        """
        return self.current_prompt
    
    def evaluate(self, generated_content, input_data):
        """
        Evaluate the generated content against the rubric.
        
        Args:
            generated_content (str): The content to evaluate.
            input_data: The original input data.
            
        Returns:
            dict: Evaluation results with keys like 'score', 'feedback', etc.
        """
        # Replace placeholders in the current prompt
        eval_prompt = self.current_prompt.replace("{input_placeholder}", str(input_data))
        eval_prompt = eval_prompt.replace("{content_placeholder}", generated_content)
        
        # Use the centralized llm_call function
        messages = [{"role": "user", "content": eval_prompt}]
        
        try:
            # Mock API key for testing purposes
            response = self.llm_call(messages=messages, model=self.model_name, openrouter_api_key=os.getenv("OPENROUTER_API_KEY"))
            
            # Parse the response
            if response and "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                
                # Parse content to extract structured data
                lines = content.strip().split('\n')
                score = 3.0  # Default score
                feedback = "No feedback provided"
                improved_output = "No improved output provided"
                summary = "No summary provided"
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("Score:"):
                        try:
                            score = float(line.replace("Score:", "").strip())
                        except:
                            pass
                    elif line.startswith("Feedback:"):
                        feedback = line.replace("Feedback:", "").strip()
                    elif line.startswith("Improved Output:"):
                        improved_output = line.replace("Improved Output:", "").strip()
                    elif line.startswith("Summary:"):
                        summary = line.replace("Summary:", "").strip()
                
                return {
                    'score': score,
                    'feedback': feedback,
                    'improved_output': improved_output,
                    'summary': summary
                }
        except Exception as e:
            print(f"Error in evaluation: {e}")
        
        # Fallback if parsing fails or an exception occurs
        return {
            'score': 3.0,
            'feedback': "Error obtaining detailed feedback",
            'improved_output': "Error obtaining improved output example",
            'summary': "Error obtaining evaluation summary"
        } 