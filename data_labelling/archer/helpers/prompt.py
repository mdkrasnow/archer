"""
This module defines the Prompt class which represents a prompt used in the Archer system.
"""

class Prompt:
    """
    Represents a prompt for LLM generation.
    
    Attributes:
        content (str): The textual content of the prompt.
        score (float, optional): Numerical score assigned to this prompt, typically by evaluation.
        feedback (str, optional): Feedback on the prompt's performance.
        generation (int, optional): The generation number of this prompt (for tracking evolution).
        history (list): List of tuples containing previous versions (content, score, feedback).
        llm_call: A function or method to call an LLM with this prompt.
    """
    
    def __init__(self, content, score=0.0, feedback_or_generation=None, generation=0):
        """
        Initialize a new Prompt instance.
        
        Args:
            content (str): The prompt text content.
            score (float, optional): Initial score for the prompt. Defaults to 0.0.
            feedback_or_generation: Can be either a feedback string or generation number.
            generation (int, optional): Initial generation number. Defaults to 0.
        """
        self.content = content
        self.score = score
        
        # Handle the third parameter which could be either feedback or generation
        if isinstance(feedback_or_generation, int):
            self.feedback = ""
            self.generation = feedback_or_generation
        else:
            self.feedback = feedback_or_generation or ""
            self.generation = generation
            
        self.history = []
        self.llm_call = None
    
    def update(self, new_content, score=None, feedback=None):
        """
        Update the prompt with new content, score, and feedback.
        Also stores the previous version in history.
        
        Args:
            new_content (str): New content for the prompt.
            score (float, optional): New score for the prompt.
            feedback (str, optional): New feedback for the prompt.
        """
        # Store the current version in history before updating
        self.history.append((self.content, self.score, self.feedback))
        
        # Update the prompt
        self.content = new_content
        if score is not None:
            self.score = score
        if feedback is not None:
            self.feedback = feedback
        self.generation += 1
    
    def __str__(self):
        """Return a human-readable string representation of the Prompt."""
        content_preview = self.content[:50] + "..." if self.content else "..."
        return f"Prompt (Gen {self.generation}, Score: {self.score:.2f}): {content_preview}"
    
    def __repr__(self):
        """Return a string representation of the Prompt."""
        return f"Prompt(content='{self.content[:30]}...', score={self.score}, generation={self.generation})" 