"""
This module defines the Model class for representing a trainable prompt-based model.

The Model class encapsulates a set of prompts that form a model which can be optimized
using AdaLflow. It provides methods for managing prompts, evaluating model performance,
and integrating with the PromptOptimizer.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import json
import copy
import adalflow as adal
from adalflow.optim.parameter import Parameter, ParameterType

from archer.helpers.prompt import Prompt
from archer.helpers.llm_call import llm_call


class Model:
    """
    A class representing a trainable prompt-based model.
    
    The Model class encapsulates a collection of prompts that can be optimized
    using AdaLflow. It provides methods for adding, removing, and evaluating
    prompts, as well as integrating with the PromptOptimizer.
    
    Attributes:
        name (str): Name of the model for identification.
        prompts (Dict[str, Prompt]): Dictionary of prompts with their identifiers.
        functions (Dict[str, Callable]): Dictionary of functions using the prompts.
        model_type (str): Type of the model (e.g., "generator", "evaluator").
        adalflow_enabled (bool): Whether AdaLflow is enabled for this model.
        version (str): Version of the model.
        metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    def __init__(self, 
                 name: str, 
                 prompts: Optional[Dict[str, Prompt]] = None,
                 functions: Optional[Dict[str, Callable]] = None,
                 model_type: str = "generator",
                 adalflow_enabled: bool = False,
                 version: str = "1.0.0",
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new Model.
        
        Args:
            name: Name of the model for identification.
            prompts: Dictionary of prompts with their identifiers (optional).
            functions: Dictionary of functions using the prompts (optional).
            model_type: Type of the model (e.g., "generator", "evaluator").
            adalflow_enabled: Whether AdaLflow is enabled for this model.
            version: Version of the model.
            metadata: Additional metadata about the model.
        """
        self.name = name
        self.prompts = prompts or {}
        self.functions = functions or {}
        self.model_type = model_type
        self.adalflow_enabled = adalflow_enabled
        self.version = version
        self.metadata = metadata or {}
        self.performance_history = []
        
        # AdaLflow parameters
        self.adalflow_params = {}
        if adalflow_enabled:
            self._init_adalflow_params()
    
    def _init_adalflow_params(self):
        """Initialize AdaLflow parameters from the current prompts."""
        for prompt_id, prompt in self.prompts.items():
            self.adalflow_params[prompt_id] = Parameter(
                data=prompt.content,
                role_desc=f"Prompt '{prompt_id}' in model '{self.name}'",
                requires_opt=True,
                param_type=ParameterType.PROMPT
            )
    
    def add_prompt(self, prompt_id: str, prompt: Prompt) -> None:
        """
        Add a prompt to the model.
        
        Args:
            prompt_id: Identifier for the prompt.
            prompt: The Prompt object to add.
        """
        self.prompts[prompt_id] = prompt
        
        # If AdaLflow is enabled, create a parameter for this prompt
        if self.adalflow_enabled:
            self.adalflow_params[prompt_id] = Parameter(
                data=prompt.content,
                role_desc=f"Prompt '{prompt_id}' in model '{self.name}'",
                requires_opt=True,
                param_type=ParameterType.PROMPT
            )
    
    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """
        Get a prompt from the model by its identifier.
        
        Args:
            prompt_id: Identifier of the prompt to retrieve.
            
        Returns:
            The Prompt object if found, None otherwise.
        """
        return self.prompts.get(prompt_id)
    
    def remove_prompt(self, prompt_id: str) -> bool:
        """
        Remove a prompt from the model.
        
        Args:
            prompt_id: Identifier of the prompt to remove.
            
        Returns:
            True if the prompt was removed, False if it wasn't found.
        """
        if prompt_id in self.prompts:
            del self.prompts[prompt_id]
            if prompt_id in self.adalflow_params:
                del self.adalflow_params[prompt_id]
            return True
        return False
    
    def add_function(self, function_id: str, function: Callable) -> None:
        """
        Add a function to the model.
        
        Args:
            function_id: Identifier for the function.
            function: The function to add.
        """
        self.functions[function_id] = function
    
    def get_function(self, function_id: str) -> Optional[Callable]:
        """
        Get a function from the model by its identifier.
        
        Args:
            function_id: Identifier of the function to retrieve.
            
        Returns:
            The function if found, None otherwise.
        """
        return self.functions.get(function_id)
    
    def remove_function(self, function_id: str) -> bool:
        """
        Remove a function from the model.
        
        Args:
            function_id: Identifier of the function to remove.
            
        Returns:
            True if the function was removed, False if it wasn't found.
        """
        if function_id in self.functions:
            del self.functions[function_id]
            return True
        return False
    
    def update_prompt(self, prompt_id: str, new_content: str, 
                     score: Optional[float] = None, 
                     feedback: Optional[str] = None) -> bool:
        """
        Update a prompt in the model.
        
        Args:
            prompt_id: Identifier of the prompt to update.
            new_content: New content for the prompt.
            score: New score for the prompt (optional).
            feedback: New feedback for the prompt (optional).
            
        Returns:
            True if the prompt was updated, False if it wasn't found.
        """
        if prompt_id in self.prompts:
            prompt = self.prompts[prompt_id]
            prompt.update(new_content=new_content, score=score, feedback=feedback)
            
            # Also update the AdaLflow parameter if enabled
            if self.adalflow_enabled and prompt_id in self.adalflow_params:
                self.adalflow_params[prompt_id].data = new_content
                
            return True
        return False
    
    def optimize_prompt(self, prompt_id: str, optimizer, feedback: str, score: float) -> bool:
        """
        Optimize a prompt in the model using the provided optimizer.
        
        Args:
            prompt_id: Identifier of the prompt to optimize.
            optimizer: The PromptOptimizer to use.
            feedback: Feedback to use for optimization.
            score: Score to use for optimization.
            
        Returns:
            True if the prompt was optimized, False if it wasn't found.
        """
        if prompt_id in self.prompts:
            prompt = self.prompts[prompt_id]
            improved_content = optimizer.optimize_prompt(prompt, feedback, score)
            self.update_prompt(prompt_id, improved_content, score, feedback)
            return True
        return False
    
    def evaluate(self, evaluator, input_data: Any) -> Dict[str, Any]:
        """
        Evaluate the model using the provided evaluator.
        
        This method generates content using the model's functions and prompts,
        then evaluates the generated content using the provided evaluator.
        
        Args:
            evaluator: The evaluator to use.
            input_data: Input data for generation.
            
        Returns:
            A dictionary containing evaluation results.
        """
        results = {}
        
        # Generate content using each function in the model
        for function_id, function in self.functions.items():
            # Call the function to generate content
            # The function should be designed to use the model's prompts
            content = function(input_data, self)
            
            # Evaluate the generated content
            evaluation = evaluator.evaluate(content, input_data)
            
            # Store the evaluation results
            results[function_id] = {
                'content': content,
                'evaluation': evaluation
            }
        
        # Calculate overall model score (e.g., average of function scores)
        overall_score = 0.0
        num_scores = 0
        
        for function_result in results.values():
            if 'evaluation' in function_result and 'score' in function_result['evaluation']:
                overall_score += function_result['evaluation']['score']
                num_scores += 1
        
        if num_scores > 0:
            overall_score /= num_scores
        
        # Add overall score to results
        results['overall_score'] = overall_score
        
        # Store performance history
        self.performance_history.append({
            'input_data': str(input_data)[:100],  # Store truncated input for reference
            'overall_score': overall_score,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        })
        
        return results
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the model.
        """
        # Convert prompts to a serializable format
        prompt_dicts = {}
        for prompt_id, prompt in self.prompts.items():
            prompt_dicts[prompt_id] = {
                'content': prompt.content,
                'score': prompt.score,
                'feedback': prompt.feedback,
                'generation': prompt.generation,
                'history': prompt.history
            }
        
        # Functions can't be serialized directly, so just store their IDs
        function_ids = list(self.functions.keys())
        
        return {
            'name': self.name,
            'model_type': self.model_type,
            'adalflow_enabled': self.adalflow_enabled,
            'version': self.version,
            'metadata': self.metadata,
            'prompts': prompt_dicts,
            'function_ids': function_ids,
            'performance_history': self.performance_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], functions: Dict[str, Callable] = None) -> 'Model':
        """
        Create a Model from a dictionary representation.
        
        Args:
            data: Dictionary representation of the model.
            functions: Dictionary of functions to attach to the model.
            
        Returns:
            A new Model instance.
        """
        # Recreate prompts from their dictionary representations
        prompts = {}
        for prompt_id, prompt_data in data.get('prompts', {}).items():
            prompt = Prompt(
                content=prompt_data['content'],
                score=prompt_data.get('score', 0.0),
                feedback_or_generation=prompt_data.get('feedback', ''),
                generation=prompt_data.get('generation', 0)
            )
            
            # Restore history if available
            if 'history' in prompt_data:
                prompt.history = prompt_data['history']
                
            prompts[prompt_id] = prompt
        
        # Create a new model instance
        model = cls(
            name=data.get('name', 'unnamed_model'),
            prompts=prompts,
            functions=functions or {},
            model_type=data.get('model_type', 'generator'),
            adalflow_enabled=data.get('adalflow_enabled', False),
            version=data.get('version', '1.0.0'),
            metadata=data.get('metadata', {})
        )
        
        # Restore performance history if available
        if 'performance_history' in data:
            model.performance_history = data['performance_history']
            
        return model
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path where the model should be saved.
        """
        model_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str, functions: Dict[str, Callable] = None) -> 'Model':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the model file.
            functions: Dictionary of functions to attach to the model.
            
        Returns:
            A new Model instance.
        """
        with open(filepath, 'r') as f:
            model_dict = json.load(f)
        
        return cls.from_dict(model_dict, functions)
    
    def clone(self) -> 'Model':
        """
        Create a deep copy of the model.
        
        Returns:
            A new Model instance that is a copy of this one.
        """
        # Create a dictionary representation (excluding functions)
        model_dict = self.to_dict()
        
        # Create a new model with the same functions
        return Model.from_dict(model_dict, self.functions) 