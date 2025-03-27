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
        self.prompts_per_generation = []
        self.scores_per_generation = []
        self.prompt_history = {}
    
    def record_generation(self, generation_num, prompts, scores=None):
        """
        Record data for a specific generation.
        
        Args:
            generation_num: The generation number
            prompts: List of prompts in this generation
            scores: Scores corresponding to the prompts (optional)
        """
        self.generations[generation_num] = prompts
        self.prompts_per_generation.append(len(prompts))
        
        # Record the prompt texts for future reference
        generation_prompts = []
        for p in prompts:
            prompt_id = id(p)
            prompt_text = p.content[:100] + "..." if len(p.content) > 100 else p.content
            generation_prompts.append((prompt_id, prompt_text))
            
            # Store in history
            if prompt_id not in self.prompt_history:
                self.prompt_history[prompt_id] = {
                    "first_seen": generation_num,
                    "content": prompt_text,
                    "generations": []
                }
            self.prompt_history[prompt_id]["generations"].append(generation_num)
        
        # Record scores if provided
        if scores:
            avg_score = sum(scores) / len(scores) if scores else 0
            self.scores_per_generation.append(avg_score)
        else:
            # If no scores provided, use -1 as a placeholder
            self.scores_per_generation.append(-1)
    
    def update_prompt_performance(self, prompts, evaluations):
        """
        Update performance metrics for prompts based on evaluation results.
        
        Args:
            prompts: List of Prompt objects
            evaluations: List of evaluation tuples (prompt, content, result)
        """
        # Extract scores from evaluations
        scores = [eval_result[2].get('score', 0.0) for eval_result in evaluations]
        
        # Calculate average score for this evaluation batch
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Record this data point
        if self.generations:
            current_gen = max(self.generations.keys()) + 1
        else:
            current_gen = 0
            
        # Record the data
        self.record_generation(current_gen, prompts, scores)
        
        # Add detailed tracking if needed
        for prompt, content, result in evaluations:
            prompt_id = id(prompt)
            score = result.get('score', 0.0)
            
            if prompt_id in self.prompt_history:
                if "scores" not in self.prompt_history[prompt_id]:
                    self.prompt_history[prompt_id]["scores"] = []
                self.prompt_history[prompt_id]["scores"].append(score)
                self.prompt_history[prompt_id]["last_seen"] = current_gen
    
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