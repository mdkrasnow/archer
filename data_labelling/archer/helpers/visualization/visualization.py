from typing import List, Dict, Any
from prompt import Prompt

class PerformanceTracker:
    """
    Tracks and visualizes the performance of the system over time.
    
    This class collects and analyzes data about the performance of prompts
    across generations, providing metrics for monitoring progress.
    """
    
    def __init__(self):
        """Initialize a new PerformanceTracker instance."""
        # List of data points for each generation
        self.generation_data = []
        
    def record_generation(self, generation: int, prompts: List[Prompt]):
        """
        Record data for a generation.
        
        Args:
            generation: The generation number
            prompts: List of Prompt objects in this generation
        """
        # Extract scores from all prompts
        scores = [p.score for p in prompts]
        
        # Create a data point for this generation
        data = {
            'generation': generation,
            'mean_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'prompt_data': [(p.content[:50], p.score) for p in prompts]
        }
        
        # Add to the collection of generation data
        self.generation_data.append(data)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get metrics summarizing system performance.
        
        Returns:
            Dictionary containing performance metrics:
            - current_generation: Current generation number
            - score_improvement: Improvement since first generation
            - best_score: Best score across all generations
            - convergence: Measure of how close to convergence (0.0-1.0)
        """
        if not self.generation_data:
            return {'status': 'No data available'}
            
        # Get latest and first generation data
        latest_gen = self.generation_data[-1]
        first_gen = self.generation_data[0]
        
        # Calculate metrics
        metrics = {
            'current_generation': latest_gen['generation'],
            'score_improvement': latest_gen['mean_score'] - first_gen['mean_score'],
            'best_score': max(g['max_score'] for g in self.generation_data),
            'convergence': self._calculate_convergence(),
        }
        return metrics
    
    def _calculate_convergence(self) -> float:
        """
        Calculate how much the scores have converged.
        
        This measures whether the system is reaching an optimum by analyzing
        the rate of improvement over recent generations.
        
        Returns:
            Convergence value between 0.0 (not converged) and 1.0 (fully converged)
        """
        if len(self.generation_data) < 3:
            return 0.0  # Not enough data for convergence analysis
            
        # Look at scores from the last few generations
        recent_scores = [g['mean_score'] for g in self.generation_data[-3:]]
        
        # If scores are identical, we have perfect convergence
        if len(set(recent_scores)) == 1:
            return 1.0
            
        # Calculate improvements between consecutive generations
        improvements = [recent_scores[i] - recent_scores[i-1] 
                        for i in range(1, len(recent_scores))]
        
        # Average improvement rate
        avg_improvement = sum(improvements) / len(improvements)
        
        # Convert to a convergence metric (0.0-1.0)
        # Smaller improvements indicate greater convergence
        return max(0.0, 1.0 - abs(avg_improvement) * 10)