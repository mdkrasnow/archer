"""
This module defines classes for visualizing and tracking performance in the Archer system.
"""

class PerformanceTracker:
    """
    A class for tracking and visualizing the performance of prompts over time.
    
    Attributes:
        generations (dict): Dictionary mapping generation numbers to prompt data.
    """
    
    def __init__(self):
        """Initialize a new PerformanceTracker."""
        self.generations = {}
    
    def record_generation(self, generation_num, prompts):
        """
        Record the state of prompts for a generation.
        
        Args:
            generation_num (int): The generation number.
            prompts (list): List of Prompt objects for this generation.
        """
        # Record the prompts and their scores
        prompt_data = []
        for prompt in prompts:
            prompt_data.append({
                'content': prompt.content,
                'score': prompt.score
            })
        
        self.generations[generation_num] = prompt_data
    
    def plot_performance(self):
        """
        Generate a plot of prompt performance over generations.
        
        Returns:
            dict: Data for visualization.
        """
        # This is a stub implementation for testing
        # In a real implementation, this would create an actual visualization
        
        # Extract data for the plot
        generations = list(self.generations.keys())
        scores = []
        
        for gen_num in generations:
            gen_scores = [p.get('score', 0) for p in self.generations[gen_num]]
            avg_score = sum(gen_scores) / len(gen_scores) if gen_scores else 0
            scores.append(avg_score)
        
        return {
            'generations': generations,
            'scores': scores
        } 