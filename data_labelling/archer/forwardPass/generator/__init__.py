"""
Generator package for the Archer system.
This module provides the GenerativeModel class for generating content with LLMs.
"""

from archer.forwardPass.generator.generator import GenerativeModel
from archer.helpers.llm_call import llm_call as default_llm_call

__all__ = ['GenerativeModel', 'default_llm_call'] 