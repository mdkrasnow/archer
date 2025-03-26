"""
Module containing Archer's backward pass components.

This includes:
- PromptOptimizer: For optimizing prompts based on feedback
- PromptEvaluator: For evaluating prompt variants to find the best performers
- Model: For representing trainable prompt-based models
- DanielsonModel: Specialization of Model for Danielson framework
"""

from backwardPass.promptOptimizer import PromptOptimizer
from backwardPass.model import Model
from backwardPass.danielson_model import DanielsonModel 