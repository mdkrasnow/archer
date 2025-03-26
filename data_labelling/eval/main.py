from typing import List, Dict, Any
from archer.forwardPass.generator import GenerativeModel
from archer.forwardPass.evaluator import AIExpert
from archer.forwardPass.human import HumanValidation
from archer.backwardPass.promptOptimizer import PromptOptimizer
from helpers.visualization import PerformanceTracker
from archer.helpers.llm_call import llm_call

class ArcherSystem:
    """
    Main class orchestrating the entire Archer feedback system.
    
    This class coordinates all components of the system, managing the
    forward and backward passes, and tracking overall performance.
    """
    
    def __init__(self, generative_model_name: str, expert_model_name: str,
                optimizer_model_name: str, initial_prompts: List[str],
                knowledge_base: List[str], rubric: Dict[str, Any]):
        """
        Initialize a new ArcherSystem.
        
        Args:
            generative_model_name: LLM to use for content generation
            expert_model_name: LLM to use for evaluation
            optimizer_model_name: LLM to use for prompt optimization
            initial_prompts: List of prompt texts to seed the population
            knowledge_base: List of documents for the expert's RAG system
            rubric: Dictionary defining evaluation criteria
        """
        # Initialize all component systems
        
        # The LLM that generates content
        self.generator = GenerativeModel(generative_model_name)
        
        # The LLM that evaluates generated content
        self.expert = AIExpert(expert_model_name, knowledge_base, rubric)
        
        # System for collecting human feedback
        self.human_validation = HumanValidation()
        
        # System for optimizing prompts
        self.optimizer = PromptOptimizer(optimizer_model_name)
                
        # System for tracking performance metrics
        self.tracker = PerformanceTracker()
        
        # Set up initial population of prompts
        self.evolution.initialize_population(initial_prompts)
        
        # Configure generator with initial prompts
        self.generator.set_prompts(self.evolution.population)

        self.llm_call = llm_call
        
    def run_forward_pass(self, input_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Run a forward pass through the system.
        
        This represents the content generation and evaluation phase.
        
        Args:
            input_data: List of inputs to generate content for
            
        Returns:
            List of results for each input-prompt combination
        """
        all_results = []
        
        # Process each input
        for data in input_data:
            # Generate content using all active prompts
            generated_outputs = self.generator.generate(data)
            
            # For each generated output
            for output, prompt in generated_outputs:
                # Have the AI expert evaluate the output
                evaluation = self.expert.evaluate(output, data)
                
                # Present to humans for validation
                validated_eval = self.human_validation.present_for_validation(
                    data, output, evaluation
                )
                
                # Save the validated evaluation
                self.human_validation.save_validation(validated_eval)
                
                # Record results for this input-prompt combination
                result = {
                    'input': data,
                    'output': output,
                    'prompt': prompt,
                    'evaluation': validated_eval
                }
                all_results.append(result)
                
        return all_results
    
    def run_backward_pass(self):
        """
        Run a backward pass to optimize prompts.
        
        This represents the learning and optimization phase.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Get all validated evaluation data
        training_data = self.human_validation.get_training_data()
        
        # Prepare to collect scores and feedback for each prompt
        prompt_scores = {}
        prompt_feedback = {}
        
        # Process each validated evaluation
        for data in training_data:
            prompt = data.get('prompt')
            if prompt:
                # Extract score and feedback
                score = data['evaluation']['score']
                feedback = data['evaluation']['feedback']
                
                # Initialize if this is the first data for this prompt
                if prompt.content not in prompt_scores:
                    prompt_scores[prompt.content] = []
                    prompt_feedback[prompt.content] = []
                    
                # Collect all scores and feedback for this prompt
                prompt_scores[prompt.content].append(score)
                prompt_feedback[prompt.content].append(feedback)
        
        # Update each prompt with its aggregated scores and feedback
        for prompt in self.evolution.population:
            if prompt.content in prompt_scores:
                # Calculate average score
                avg_score = sum(prompt_scores[prompt.content]) / len(prompt_scores[prompt.content])
                
                # Combine all feedback with separators
                combined_feedback = " | ".join(prompt_feedback[prompt.content])
                
                # Update the prompt
                prompt.score = avg_score
                prompt.feedback = combined_feedback
        
        # Select the best-performing prompts to survive
        survivors = self.evolution.select_survivors()
        
        # Create the next generation of prompts
        next_gen = self.evolution.create_next_generation(survivors, self.optimizer)
        
        # Update the generator with the new prompts
        self.generator.set_prompts(next_gen)
        
        # Record this generation's performance
        self.tracker.record_generation(self.evolution.generation, next_gen)
        
        # Return performance metrics
        return self.tracker.get_performance_metrics()
    
    def run_iteration(self, input_data: List[Any]):
        """
        Run a complete iteration of the system.
        
        This includes both forward and backward passes.
        
        Args:
            input_data: List of inputs to generate content for
            
        Returns:
            Dictionary containing forward results and performance metrics
        """
        # Run forward pass (generation and evaluation)
        forward_results = self.run_forward_pass(input_data)
        
        # Run backward pass (learning and optimization)
        backward_metrics = self.run_backward_pass()
        
        # Return combined results
        return {
            'forward_results': forward_results,
            'performance_metrics': backward_metrics
        }