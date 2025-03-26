"""
Archer.py

This module defines the Archer class which orchestrates the overall
feedback system for optimizing prompts. It integrates the forward pass
(generation and evaluation) and the backward pass (prompt optimization)
to create a loop that continually refines the prompts for a given task.

Note:
- The knowledge_base parameter is now a list of directory paths. All text files 
  found in these directories will be loaded as documents.
- The rubric is now a string containing the evaluation criteria.
"""

import os
import random
from typing import List, Dict, Any, Callable, Union, Tuple, Optional
from .helpers.prompt import Prompt
from .backwardPass.promptOptimizer import PromptOptimizer
from .backwardPass.PromptEvaluator.promptEvaluator import PromptEvaluator
from .forwardPass.evaluator import AIExpert
from .forwardPass.generator import GenerativeModel
from .forwardPass.human.human import HumanValidation
from .helpers.visualization import PerformanceTracker
from .helpers.logging_utils import get_logger, log_entry_exit, log_call_args
# Assuming we'll implement Argilla integration, import placeholder:
# from database.argilla import ArgillaDB

# Setup logger for this module
logger = get_logger(__name__)

@log_entry_exit(logger)
def load_knowledge_from_directories(directories: list) -> list:
    """
    Load all text documents from a list of directories.

    Args:
        directories: List of directory paths as strings.

    Returns:
        A list of document contents (strings).
    """
    logger.info(f"Loading knowledge from {len(directories)} directories")
    documents = []
    for directory in directories:
        if os.path.isdir(directory):
            files_processed = 0
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            documents.append(f.read())
                            files_processed += 1
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {e}")
            logger.info(f"Loaded {files_processed} files from directory: {directory}")
        else:
            logger.warning(f"Directory not found: {directory}")
    logger.info(f"Finished loading {len(documents)} knowledge documents")
    return documents


class Archer:
    def __init__(self,
                 generator_model_name: str,
                 evaluator_model_name: str,
                 optimizer_model_name: str,
                 knowledge_base: list,   # List of directory paths
                 rubric: str,            # Rubric as a string
                 initial_prompts: list,
                 openrouter_api_key: str,
                 input_spec: Union[str, List[str]] = "string",
                 output_spec: str = "string",
                 evaluation_fields: list = None,
                 input_types: List[str] = None,
                 resampling_enabled: bool = True,
                 input_interaction_mode: str = "parallel",
                 validation_attempts_per_param: int = 5,
                 top_params_percentile: float = 0.25,
                 variation_traits: List[str] = None,
                 max_prompts_per_cycle: int = 4,
                 # New parameters for additional functionality
                 human_validation_enabled: bool = False,
                 num_simulations_per_prompt: int = 3,
                 database_config: Optional[Dict[str, Any]] = None,
                 adalflow_enabled: bool = False,
                 adalflow_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new Archer instance.

        Args:
            generator_model_name: Identifier for the generative model (e.g., "gpt-4").
            evaluator_model_name: Identifier for the evaluator model (e.g., "claude-3").
            optimizer_model_name: Identifier for the LLM used in prompt optimization.
            knowledge_base: List of directory paths containing text documents for RAG.
            rubric: A string defining the evaluation criteria.
            initial_prompts: List of Prompt objects to initialize the system.
            openrouter_api_key: API key for making LLM calls.
            input_spec: Specification(s) for the input data (default: "string").
                        Can be a single string or list of strings for multiple inputs.
            output_spec: Specification for the output data (default: "string").
            evaluation_fields: List of fields expected in evaluation (default:
                               ['score', 'feedback', 'improved_output', 'summary']).
            input_types: List of data types for each input source.
            resampling_enabled: Whether to allow resampling of inputs.
            input_interaction_mode: How multiple inputs interact ('parallel' or 'combinatorial').
            validation_attempts_per_param: Number of validation runs per parameter set.
            top_params_percentile: Percentile of top-performing parameters to keep.
            variation_traits: List of traits that cause variations in scores.
            max_prompts_per_cycle: Maximum number of prompts to use in each cycle.
            human_validation_enabled: Whether to enable human validation in the process.
            num_simulations_per_prompt: Number of simulations to run per prompt in the PromptEvaluator.
            database_config: Configuration for the database (Argilla) integration.
            adalflow_enabled: Whether to use AdaLflow for prompt optimization.
            adalflow_config: Configuration for AdaLflow integration.
        """
        logger.info("Initializing Archer system")
        
        if evaluation_fields is None:
            evaluation_fields = ['score', 'feedback', 'improved_output', 'summary']
        
        if input_types is None:
            input_types = ["string"]
        
        if variation_traits is None:
            variation_traits = []

        # Basic configuration
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.evaluation_fields = evaluation_fields
        self.openrouter_api_key = openrouter_api_key
        
        # Advanced configuration for input handling
        self.input_types = input_types
        self.resampling_enabled = resampling_enabled
        self.input_interaction_mode = input_interaction_mode
        
        # Backward pass configuration
        self.validation_attempts_per_param = validation_attempts_per_param
        self.top_params_percentile = top_params_percentile
        self.variation_traits = variation_traits
        self.max_prompts_per_cycle = max_prompts_per_cycle
        
        # New configurations
        self.human_validation_enabled = human_validation_enabled
        self.adalflow_enabled = adalflow_enabled
        self.adalflow_config = adalflow_config if adalflow_config else {}
        self.database_config = database_config if database_config else {}
        self.num_simulations_per_prompt = num_simulations_per_prompt

        logger.info(f"Generator model: {generator_model_name}")
        logger.info(f"Evaluator model: {evaluator_model_name}")
        logger.info(f"Optimizer model: {optimizer_model_name}")
        logger.info(f"Input spec: {input_spec}")
        logger.info(f"Input interaction mode: {input_interaction_mode}")
        logger.info(f"Human validation enabled: {human_validation_enabled}")
        logger.info(f"AdaLFlow enabled: {adalflow_enabled}")

        # Load knowledge documents from the provided directories
        knowledge_documents = load_knowledge_from_directories(knowledge_base)
        logger.info(f"Loaded {len(knowledge_documents)} knowledge documents total")

        # Initialize core components
        logger.info("Initializing generator")
        self.generator = GenerativeModel(model_name=generator_model_name, temperature=0.7)
        
        logger.info("Initializing evaluator")
        self.evaluator = AIExpert(model_name=evaluator_model_name,
                                  knowledge_base=knowledge_documents,
                                  rubric=rubric)
        
        # Initialize the optimizer with AdaLFlow support if enabled
        logger.info("Initializing optimizer")
        adalflow_optimizer_config = adalflow_config or {}
        self.optimizer = PromptOptimizer(
            model_name=optimizer_model_name,
            temperature=adalflow_optimizer_config.get('temperature', 0.7),
            adalflow_enabled=adalflow_enabled,
            max_trials=adalflow_optimizer_config.get('max_trials', 5),
            top_k=adalflow_optimizer_config.get('top_k', 3)
        )
        
        # Store AdaLFlow configuration
        self.adalflow_enabled = adalflow_enabled
        self.adalflow_config = adalflow_config or {}
        
        logger.info("Initializing performance tracker")
        self.performance_tracker = PerformanceTracker()
        
        # Initialize new components
        if human_validation_enabled:
            logger.info("Initializing human validator")
            self.human_validator = HumanValidation()
        else:
            self.human_validator = None
        
        # Initialize prompt evaluator
        logger.info("Initializing prompt evaluator")
        self.prompt_evaluator = PromptEvaluator(
            generative_model=self.generator,
            evaluator=self.evaluator,
            num_simulations=self.num_simulations_per_prompt,
            quantile_threshold=self.top_params_percentile
        )
        
        # Initialize database if config is provided
        # if database_config:
        #     logger.info("Initializing database connection")
        #     self.database = ArgillaDB(**database_config)
        # else:
        #     self.database = None

        # Set the active prompts in the generator
        self.active_prompts = initial_prompts
        logger.info(f"Setting {len(initial_prompts)} initial prompts")
        self.generator.set_prompts(self.active_prompts)

        self.generation_count = 0
        
        # Store candidate prompts for evaluation
        self.candidate_prompts = []
        logger.info("Archer system initialization complete")

    @log_entry_exit(logger)
    def run_forward_pass(self, input_data: Any) -> list:
        """
        Runs the forward pass: generates content using active prompts and evaluates it.

        Args:
            input_data: The input data to feed into the generator.
                        Can be a single item or a list/tuple for multiple inputs.

        Returns:
            A list of tuples: (Prompt, generated content, evaluation result dict).
        """
        logger.info(f"Starting forward pass with generation count {self.generation_count}")
        
        # Handle multiple input types
        if isinstance(input_data, (list, tuple)) and isinstance(self.input_spec, list):
            # Handle the input based on interaction mode
            if self.input_interaction_mode == "parallel":
                # Zip inputs together - each element corresponds to one row
                input_rows = zip(*input_data)
                logger.info(f"Using parallel input mode with {len(input_data[0]) if input_data else 0} rows")
            else:  # combinatorial
                # Create all possible combinations
                from itertools import product
                input_rows = product(*input_data)
                # We can't know the length without consuming the iterator, so just log the mode
                logger.info(f"Using combinatorial input mode")
        else:
            # Single input type
            input_rows = [input_data]
            logger.info("Using single input mode")
        
        # Limit the number of active prompts per cycle
        active_prompts = self.active_prompts[:self.max_prompts_per_cycle]
        logger.info(f"Using {len(active_prompts)} active prompts out of {len(self.active_prompts)} total")
        self.generator.set_prompts(active_prompts)
        
        all_evaluations = []
        
        for input_row in input_rows:
            logger.info(f"Processing input: {input_row if isinstance(input_row, str) else '(complex input)'}")
            generated_outputs = self.generator.generate(input_row)
            logger.info(f"Generated {len(generated_outputs)} outputs")
            
            for content, prompt in generated_outputs:
                logger.info(f"Evaluating output for prompt: {prompt.content[:50]}...")
                eval_result = self.evaluator.evaluate(
                    generated_content=content, 
                    input_data=input_row
                )
                
                # Log the evaluation score
                score = eval_result.get('score', 'N/A')
                logger.info(f"Evaluation score: {score}")
                
                # If human validation is enabled, present for validation
                if self.human_validation_enabled and self.human_validator:
                    logger.info("Presenting evaluation for human validation")
                    eval_result = self.human_validator.present_for_validation(
                        input_data=input_row,
                        generated_content=content,
                        ai_evaluation=eval_result
                    )
                    # Save the validated evaluation for later analysis
                    logger.info("Saving human-validated evaluation")
                    self.human_validator.save_validation(eval_result)
                
                all_evaluations.append((prompt, content, eval_result))

        logger.info(f"Recording generation {self.generation_count} performance")
        self.performance_tracker.record_generation(self.generation_count, active_prompts)
        
        # Store evaluations in database if available
        # if hasattr(self, 'database') and self.database:
        #     logger.info(f"Saving {len(all_evaluations)} evaluations to database")
        #     self.database.save_evaluations(self.generation_count, all_evaluations)
        
        logger.info(f"Forward pass complete with {len(all_evaluations)} total evaluations")
        return all_evaluations

    @log_entry_exit(logger)
    def run_backward_pass(self, evaluations: list) -> None:
        """
        Runs the backward pass: optimizes each prompt using the evaluation feedback.
        Updates the prompt content and score, then sets the new prompts for the next cycle.

        Args:
            evaluations: List of tuples (Prompt, generated content, evaluation dict).
        """
        logger.info(f"Starting backward pass with {len(evaluations)} evaluations")
        
        # Create maps for feedback and scores
        feedback_map = {}
        score_map = {}
        
        # Extract feedback and scores from evaluations
        for i, (prompt, generated_content, eval_result) in enumerate(evaluations):
            prompt_id = str(i)
            feedback_map[prompt_id] = eval_result.get('feedback', '')
            score_map[prompt_id] = eval_result.get('score', 0)
            logger.debug(f"Prompt {i} score: {score_map[prompt_id]}")
        
        # Get the prompts from evaluations
        prompts = [prompt for prompt, _, _ in evaluations]
        
        if self.adalflow_enabled:
            logger.info("Using AdaLFlow-based optimization")
            # Use AdaLFlow-based optimization
            improved_prompts = self.optimizer.optimize(
                prompt_objs=prompts,
                feedback_map=feedback_map,
                score_map=score_map
            )
            logger.info(f"AdaLFlow optimization produced {len(improved_prompts)} improved prompts")
            
            # If using a Model with adalflow, optimize the model prompts directly
            if hasattr(self, 'model') and getattr(self, 'model', None) and getattr(self.model, 'adalflow_enabled', False):
                logger.info("Optimizing model prompts with AdaLFlow")
                # Create model-specific feedback and score maps
                model_feedback_map = {}
                model_score_map = {}
                
                for prompt_id, prompt in self.model.prompts.items():
                    # Find this prompt in our evaluations if possible
                    for i, (eval_prompt, _, eval_result) in enumerate(evaluations):
                        if prompt.content == eval_prompt.content:
                            model_feedback_map[prompt_id] = eval_result.get('feedback', '')
                            model_score_map[prompt_id] = eval_result.get('score', 0)
                            break
                
                # Optimize the model prompts directly
                logger.info(f"Optimizing model with {len(model_feedback_map)} feedback items")
                self.optimizer.optimize_model(
                    model=self.model,
                    feedback_map=model_feedback_map,
                    score_map=model_score_map
                )
        else:
            # Standard optimization approach
            logger.info("Using standard prompt optimization")
            
            # For each prompt, optimize based on feedback and score
            improved_prompts = []
            improved_contents = set()  # Track unique contents to avoid duplicates
            
            # Process each prompt with its feedback and score
            for i, prompt in enumerate(prompts):
                prompt_id = str(i)
                feedback = feedback_map.get(prompt_id, '')
                score = score_map.get(prompt_id, 0)
                
                logger.info(f"Optimizing prompt #{i} with score {score}")
                
                # Get improved content through the optimizer
                improved_content = self.optimizer.optimize_prompt(prompt, feedback, score)
                
                if improved_content and improved_content not in improved_contents:
                    # Create a new prompt with the improved content
                    improved_prompt = Prompt(content=improved_content)
                    improved_prompt.parent = prompt
                    improved_prompts.append(improved_prompt)
                    improved_contents.add(improved_content)
                    
                    logger.debug(f"Created improved prompt: {improved_content[:50]}...")
            
            logger.info(f"Created {len(improved_prompts)} improved prompts through standard optimization")
            
            # Add original prompts to mix as well (they compete with the new ones)
            all_prompt_contents = improved_contents.copy()
            for prompt in prompts:
                if prompt.content not in all_prompt_contents:
                    improved_prompts.append(prompt)
                    all_prompt_contents.add(prompt.content)
            
            # Generate more variants for the prompts that performed well
            self.candidate_prompts = improved_prompts.copy()
            
            # Generate more variants for the top-performing prompts
            logger.info("Generating additional prompt variants")
            variants = self._generate_prompt_variants(improved_prompts)
            
            # Add variants to the candidate pool
            for variant in variants:
                if variant.content not in all_prompt_contents:
                    self.candidate_prompts.append(variant)
                    all_prompt_contents.add(variant.content)
            
            logger.info(f"Generated {len(variants)} additional prompt variants")
        
        # Use PromptEvaluator to evaluate all candidate prompts
        if hasattr(self, 'prompt_evaluator') and self.prompt_evaluator:
            logger.info("Evaluating candidate prompts with PromptEvaluator")
            # Generate sample input data for evaluation
            eval_inputs = self._generate_evaluation_inputs(self.validation_attempts_per_param)
            logger.info(f"Generated {len(eval_inputs)} evaluation inputs")
            
            # Evaluate prompts using the dedicated evaluator
            logger.info(f"Evaluating {len(self.candidate_prompts)} candidate prompts")
            evaluation_results = self.prompt_evaluator.evaluate_prompts(
                prompts=self.candidate_prompts, 
                input_data=eval_inputs,
                num_simulations=self.validation_attempts_per_param
            )
            
            # Update prompt scores based on evaluation results
            for prompt, avg_score, _ in evaluation_results:
                prompt.score = avg_score
                logger.debug(f"Prompt evaluation score: {avg_score} for '{prompt.content[:30]}...'")
        else:
            # Fallback to old method if prompt_evaluator is not available
            logger.info("No PromptEvaluator available, using fallback evaluation")
            self._evaluate_prompt_candidates(skip_scored_prompts=True)
        
        # Select top-performing prompts for the next cycle
        logger.info("Selecting top-performing prompts for next cycle")
        self.active_prompts = self._select_top_prompts()
        logger.info(f"Selected {len(self.active_prompts)} prompts for next cycle")
        
        # Update generator with selected prompts
        logger.info("Updating generator with selected prompts")
        self.generator.set_prompts(self.active_prompts)
        
        self.generation_count += 1
        logger.info(f"Backward pass complete, generation count now {self.generation_count}")

    @log_entry_exit(logger)
    def _generate_prompt_variants(self, base_prompts: List[Prompt]) -> List[Prompt]:
        """
        Generate variants of the best prompts by asking the optimizer to create variations.
        
        Args:
            base_prompts: List of prompt objects to generate variants from
            
        Returns:
            List of new Prompt objects representing variants
        """
        logger.info(f"Generating prompt variants from {len(base_prompts)} base prompts")
        variants = []
        
        # For each prompt, create 2 variations
        for i, prompt in enumerate(base_prompts):
            logger.debug(f"Generating variants for base prompt #{i}: {prompt.content[:40]}...")
            
            for j in range(2):  # Create 2 variations per prompt
                # Create a variation message that introduces random factors
                traits = self.variation_traits
                if not traits:
                    traits = ["clarity", "creativity", "specificity", "directness"]
                
                # Select random traits to focus on
                num_traits = min(2, len(traits))
                selected_traits = random.sample(traits, num_traits)
                
                # Create a message asking for a variation focusing on those traits
                traits_str = " and ".join(selected_traits)
                variation_message = f"Create a variation of this prompt that improves {traits_str}."
                
                logger.debug(f"Variation #{j+1} focusing on traits: {traits_str}")
                
                # Get a variation through the optimizer
                variation = self.optimizer.create_prompt_variation(
                    prompt.content, variation_message
                )
                
                if variation:
                    # Create a new prompt with this content
                    variant_prompt = Prompt(content=variation)
                    variant_prompt.parent = prompt
                    variants.append(variant_prompt)
                    logger.debug(f"Created variant: {variation[:40]}...")
        
        logger.info(f"Generated {len(variants)} total variants")
        return variants

    @log_entry_exit(logger)
    def _evaluate_prompt_candidates(self, skip_scored_prompts=False) -> None:
        """
        Evaluate all candidate prompts by simulating their performance.
        This is a fallback method used when PromptEvaluator is not available.
        
        Args:
            skip_scored_prompts: If True, skip prompts that already have scores
        """
        logger.info(f"Evaluating {len(self.candidate_prompts)} candidate prompts")
        
        # Generate sample inputs for validation
        sample_inputs = self._generate_evaluation_inputs(self.validation_attempts_per_param)
        logger.info(f"Generated {len(sample_inputs)} sample inputs for validation")
        
        # Process each candidate prompt
        for prompt in self.candidate_prompts:
            # Skip if already scored and flag is set
            if skip_scored_prompts and prompt.score is not None:
                logger.debug(f"Skipping already scored prompt: {prompt.content[:40]}...")
                continue
                
            scores = []
            
            # For each input, simulate running the prompt and getting an evaluation
            for input_data in sample_inputs:
                logger.debug(f"Simulating prompt on input: {str(input_data)[:40]}...")
                
                # Format the prompt with this input
                formatted_prompt = prompt.format(input_data)
                
                # Generate content using LLM
                generated_content = self.generator._call_llm(formatted_prompt)
                
                # Evaluate the generated content
                evaluation = self.evaluator.evaluate(generated_content, input_data)
                
                # Extract score
                score = evaluation.get('score', 0)
                scores.append(score)
                logger.debug(f"Simulation score: {score}")
            
            # Calculate average score for this prompt
            if scores:
                avg_score = sum(scores) / len(scores)
                prompt.score = avg_score
                logger.info(f"Prompt evaluated with average score: {avg_score}")
        
        logger.info("Prompt evaluation complete")

    @log_entry_exit(logger)
    def _generate_evaluation_inputs(self, count: int) -> List[Any]:
        """
        Generate random inputs for evaluating prompts.
        
        In a real system, these would be sampled from a training dataset.
        For now, we'll either create random strings or reuse the validation dataset.
        
        Args:
            count: Number of inputs to generate
            
        Returns:
            List of input data items
        """
        logger.info(f"Generating {count} evaluation inputs")
        
        # For real implementation, sample from actual data
        # For now, just create some random strings
        inputs = []
        for i in range(count):
            # This is a placeholder, actual implementation would use real data
            dummy_input = f"Evaluation input {i} for testing"
            inputs.append(dummy_input)
        
        logger.info(f"Generated {len(inputs)} evaluation inputs")
        return inputs

    @log_entry_exit(logger)
    def _select_top_prompts(self) -> List[Prompt]:
        """
        Select the top-performing prompts based on their scores.
        
        Returns:
            List of the highest-scoring prompts
        """
        logger.info("Selecting top-performing prompts")
        
        # Sort all candidate prompts by score
        sorted_prompts = sorted(
            self.candidate_prompts, 
            key=lambda p: p.score if p.score is not None else 0, 
            reverse=True
        )
        
        # Calculate how many to keep
        keep_count = max(
            1,  # Always keep at least one prompt
            min(
                round(len(sorted_prompts) * self.top_params_percentile),
                self.max_prompts_per_cycle
            )
        )
        
        # Select the top performers
        top_prompts = sorted_prompts[:keep_count]
        
        logger.info(f"Selected {len(top_prompts)} top prompts out of {len(sorted_prompts)} candidates")
        for i, prompt in enumerate(top_prompts):
            logger.debug(f"Top prompt #{i+1} score: {prompt.score}")
        
        return top_prompts

    @log_entry_exit(logger)
    def run_training_cycle(self, input_data: Any) -> list:
        """
        Run a complete training cycle (forward and backward pass).
        
        Args:
            input_data: The input data to use for this cycle
            
        Returns:
            The evaluations from the forward pass
        """
        logger.info(f"Starting training cycle {self.generation_count}")
        
        # Run forward pass
        evaluations = self.run_forward_pass(input_data)
        
        # Run backward pass
        self.run_backward_pass(evaluations)
        
        logger.info(f"Completed training cycle {self.generation_count-1}")
        return evaluations

    @log_entry_exit(logger)
    def run_training_loop(self, input_data_generator: Callable, num_cycles: int = 5) -> None:
        """
        Run multiple training cycles with different input data.
        
        Args:
            input_data_generator: Function that returns new input data for each cycle
            num_cycles: Number of training cycles to run
        """
        logger.info(f"Starting training loop with {num_cycles} cycles")
        
        for i in range(num_cycles):
            logger.info(f"Training cycle {i+1}/{num_cycles}")
            
            # Get input data for this cycle
            input_data = input_data_generator()
            logger.info(f"Obtained new input data for cycle {i+1}")
            
            # Run a training cycle
            self.run_training_cycle(input_data)
            
            # Log the current best prompts after each cycle
            best_prompts = self._select_top_prompts()
            logger.info(f"Best prompt after cycle {i+1}: {best_prompts[0].content[:50]}... (score: {best_prompts[0].score})")
        
        logger.info(f"Training loop completed after {num_cycles} cycles")

    @log_entry_exit(logger)
    def _evaluate_and_select_best_prompts(self, prompts: List[Prompt]) -> List[Prompt]:
        """
        Evaluate a set of prompts and select the best ones.
        
        This is a helper method that combines evaluation and selection in one step.
        
        Args:
            prompts: List of prompts to evaluate
            
        Returns:
            List of selected best prompts
        """
        logger.info(f"Evaluating and selecting best prompts from {len(prompts)} candidates")
        
        # Set up candidate prompts
        self.candidate_prompts = prompts
        
        # Evaluate them
        if hasattr(self, 'prompt_evaluator') and self.prompt_evaluator:
            logger.info("Using PromptEvaluator for prompt evaluation")
            eval_inputs = self._generate_evaluation_inputs(self.validation_attempts_per_param)
            evaluation_results = self.prompt_evaluator.evaluate_prompts(
                prompts=self.candidate_prompts, 
                input_data=eval_inputs,
                num_simulations=self.validation_attempts_per_param
            )
            
            for prompt, avg_score, _ in evaluation_results:
                prompt.score = avg_score
                logger.debug(f"Prompt evaluation score: {avg_score}")
        else:
            logger.info("Using fallback evaluation method")
            self._evaluate_prompt_candidates()
            
        # Select the best ones
        best_prompts = self._select_top_prompts()
        logger.info(f"Selected {len(best_prompts)} best prompts")
        
        return best_prompts