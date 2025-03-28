"""
This module provides a PromptOptimizer class for improving prompts based on evaluation feedback.
It can use simple LLM-based optimization or AdaLFlow gradient-based optimization.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import copy
import random
import logging
import time
import concurrent.futures
from functools import partial

from archer.helpers.llm_call import llm_call
from archer.helpers.prompt import Prompt

# Check if AdaLFlow is available
try:
    from adalflow.optim.parameter import Parameter, ParameterType
    from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer
    from adalflow.core import Generator
    from adalflow.components.model_client.openai_client import OpenAIClient
    ADALFLOW_AVAILABLE = True
except ImportError:
    ADALFLOW_AVAILABLE = False
    
    # Create proper mock classes for testing that are actual types (classes)
    class Parameter:
        def __init__(self, **kwargs):
            self.data = kwargs.get('data', '')
            self.role_desc = kwargs.get('role_desc', '')
            self.score = kwargs.get('score', 0.0)
            self.requires_opt = kwargs.get('requires_opt', True)
            self.param_type = kwargs.get('param_type', None)
            
        def add_gradient(self, gradient):
            pass
            
        def backward(self):
            pass
    
    class ParameterType:
        PROMPT = "PROMPT"
    
    class TGDOptimizer:
        def __init__(self, **kwargs):
            self.max_trials = kwargs.get('max_trials', 5)
            self.top_k = kwargs.get('top_k', 3)
            self.temperature = kwargs.get('temperature', 0.7)
            
        def set_parameters(self, parameters):
            pass
            
        def propose(self):
            pass
            
        def step(self):
            pass
            
    class Generator:
        def __init__(self, *, model_client=None, model_kwargs=None):
            self.model_client = model_client
            self.model_kwargs = model_kwargs or {}
            
    class OpenAIClient:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get('api_key', 'test_api_key')

class PromptOptimizer:
    """
    A class for optimizing prompts based on evaluation feedback.
    
    Can use either simple LLM-based optimization or AdaLFlow gradient-based optimization.
    The optimizer can generate variants of prompts with natural variation and integrate
    with the promptEvaluator to identify the best performing prompts.
    """
    
    def __init__(self, model_name, temperature=0.7, adalflow_enabled=False, max_trials=5, top_k=3,
                 openrouter_api_key="test_api_key", variation_traits=None):
        """
        Initialize a new PromptOptimizer.
        
        Args:
            model_name (str): Name/identifier of the LLM to use.
            temperature (float, optional): Temperature parameter for generation. Defaults to 0.7.
            adalflow_enabled (bool, optional): Whether to use AdaLFlow for optimization. Defaults to False.
            max_trials (int, optional): Maximum number of trials for TGDOptimizer. Defaults to 5.
            top_k (int, optional): Number of top candidates to keep for TGDOptimizer. Defaults to 3.
            openrouter_api_key (str, optional): API key for OpenRouter. Defaults to "test_api_key".
            variation_traits (List[str], optional): List of traits to emphasize in prompt variations.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.adalflow_enabled = adalflow_enabled and ADALFLOW_AVAILABLE
        self.max_trials = max_trials
        self.top_k = top_k
        self.openrouter_api_key = openrouter_api_key
        self.variation_traits = variation_traits or []
        
        # Reference to llm_call for the test to verify
        self.llm_call = llm_call
        
        # Initialize AdaLFlow components if enabled
        if self.adalflow_enabled:
            self._init_adalflow()
        else:
            self.generator = None
            self.optimizer = None
    
    def _init_adalflow(self):
        """Initialize AdaLFlow components properly."""
        if self.adalflow_enabled and ADALFLOW_AVAILABLE:
            try:
                # Create a more configurable generator
                self.generator = Generator(
                    model_client=OpenAIClient(api_key=self.openrouter_api_key),
                    model_kwargs={
                        "model": self.model_name,
                        "temperature": self.temperature,
                        "max_tokens": 1000,  # Configurable
                    },
                )
                
                # Configure TGD Optimizer with more options
                self.optimizer = TGDOptimizer(
                    max_trials=self.max_trials,
                    top_k=self.top_k,
                    temperature=self.temperature,
                )
            except Exception as e:
                print(f"Error initializing AdaLFlow components: {e}")
                # Fallback to safe defaults with mock instances
                mock_client = OpenAIClient(api_key=self.openrouter_api_key)
                self.generator = Generator(model_client=mock_client)
                self.optimizer = TGDOptimizer(
                    max_trials=self.max_trials,
                    top_k=self.top_k,
                    temperature=self.temperature
                )
        else:
            self.generator = None
            self.optimizer = None
    
    def _safe_adalflow_operation(self, operation_func, *args, **kwargs):
        """
        Safely execute an AdaLFlow operation with error handling.
        
        Args:
            operation_func: The AdaLFlow function to call.
            *args, **kwargs: Arguments to pass to the function.
            
        Returns:
            The result of the operation, or None if an error occurred.
        """
        if not (self.adalflow_enabled and ADALFLOW_AVAILABLE):
            return None
            
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            print(f"AdaLFlow operation failed: {e}")
            return None
    
    def _calculate_gradient_magnitude(self, score, max_score=5.0):
        """
        Calculate gradient magnitude based on score.
        
        Lower scores result in larger magnitude (more aggressive changes).
        Higher scores result in smaller magnitude (more conservative changes).
        
        Args:
            score: The evaluation score.
            max_score: The maximum possible score.
            
        Returns:
            A float representing the gradient magnitude (0.0 to 1.0).
        """
        # Normalize score to 0-1 range and invert
        normalized_score = score / max_score
        # Apply a very strong non-linear transformation to emphasize differences
        # Low scores (close to 0) will result in magnitudes close to 1.0
        # High scores (close to max_score) will result in magnitudes close to 0.1
        # For a score of 1.0 out of 5.0, the magnitude will be about 0.82
        return 0.1 + 0.9 * pow(1 - normalized_score, 1.2)
    
    def optimize_prompt(self, prompt, feedback, score):
        """
        Optimize a single prompt based on evaluation feedback using standard LLM approach.
        
        Args:
            prompt: The Prompt object to optimize.
            feedback (str): Feedback from evaluation.
            score (float): Score from evaluation.
            
        Returns:
            str: The improved prompt content.
        """
        try:
            # Calculate gradient magnitude to adjust temperature
            magnitude = self._calculate_gradient_magnitude(score)
            adjusted_temp = min(0.9, self.temperature + magnitude * 0.3)
            
            # Construct the optimization prompt with a more detailed instruction
            # based on the score and gradient magnitude
            score_guidance = (
                "This prompt needs significant improvement. Be creative and consider "
                "a substantial rewrite focusing on clarity and specificity."
                if score < 3.0 else 
                "This prompt needs moderate improvement while maintaining its core strengths."
                if score < 4.0 else
                "This prompt is already quite good. Make minor refinements to perfect it."
            )
            
            optimization_prompt = (
                f"Improve the following prompt based on feedback.\n\n"
                f"Original Prompt: {prompt.content}\n\n"
                f"Feedback: {feedback}\n"
                f"Score: {score} out of 5.\n\n"
                f"Guidance: {score_guidance}\n\n"
                "Improved Prompt:"
            )
            
            # Call LLM to generate an improved prompt
            messages = [{"role": "user", "content": optimization_prompt}]
            response = llm_call(
                messages=messages,
                model=self.model_name,
                temperature=adjusted_temp,
                openrouter_api_key=self.openrouter_api_key
            )
            
            # Extract the generated content
            improved_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # If we got a valid response, return it
            if improved_content:
                return improved_content
            
            # Fallback to original content if response is empty
            return prompt.content
            
        except Exception as e:
            # In case of any error, return the original prompt
            print(f"Error in optimize_prompt: {e}")
            return prompt.content
    
    def generate_prompt_variants(self, base_prompts, variation_traits=None, num_variants=3, max_retries=2):
        """
        Generate multiple variants of base prompts with natural variation and retry logic.
        
        Args:
            base_prompts: List of Prompt objects to use as base.
            variation_traits: List of traits to emphasize in variations.
            num_variants: Number of variants to generate per base prompt.
            max_retries: Maximum number of retries per variant.
            
        Returns:
            List of new Prompt objects (variants).
        """
        logger = logging.getLogger(__name__)
        variants = []
        traits = variation_traits or self.variation_traits
        
        # Create variation instructions
        variation_instructions = ""
        if traits:
            variation_instructions = (
                "Create variations that emphasize the following traits: " + 
                ", ".join(traits) + ". "
            )
        
        # Track generated variants to ensure we don't exceed limits
        variant_count = 0
        max_variants = len(base_prompts) * num_variants * 2  # Absolute maximum to prevent runaway generation
        
        logger.info(f"Generating up to {num_variants} variants for {len(base_prompts)} base prompts")
        
        for base_prompt in base_prompts:
            # Limit variants per base prompt
            prompt_variants = 0
            
            if variant_count >= max_variants:
                logger.warning(f"Reached maximum variant limit ({max_variants}), stopping generation")
                break
                
            for i in range(num_variants):
                if variant_count >= max_variants:
                    break
                    
                # Attempt to generate with retries
                for retry in range(max_retries + 1):
                    # Exit early if we've hit the maximum
                    if variant_count >= max_variants:
                        break
                        
                    try:
                        # Add small incremental temperature boost for variety and retries
                        temp_boost = i * 0.1 + retry * 0.05
                        
                        if self.adalflow_enabled and ADALFLOW_AVAILABLE and retry == 0:
                            try:
                                # Attempt AdaLFlow generation on first try only
                                logger.debug(f"Attempting AdaLFlow generation for prompt variant {i+1}")
                                param = Parameter(
                                    data=base_prompt.content,
                                    role_desc=f"Base prompt",
                                    requires_opt=True,
                                    param_type=ParameterType.PROMPT,
                                    score=base_prompt.score or 0.0
                                )
                                
                                # Add gradient based on feedback and score
                                param.add_gradient({
                                    "score": base_prompt.score or 0.0,
                                    "feedback": base_prompt.feedback or "",
                                    "variation_traits": traits
                                })
                                
                                # Run backward pass to analyze feedback
                                param.backward()
                                
                                # Generate variants
                                self.optimizer.set_parameters([param])
                                self.optimizer.propose()
                                self.optimizer.step()
                                
                                # Use standard LLM approach to extract the actual variant
                                variant = self._generate_variant_with_llm(
                                    base_prompt, 
                                    variation_instructions, 
                                    temperature_boost=temp_boost
                                )
                                
                                if variant:
                                    variants.append(variant)
                                    prompt_variants += 1
                                    variant_count += 1
                                    break  # Success, no need for more retries
                                    
                            except Exception as e:
                                logger.error(f"AdaLFlow variant generation failed: {str(e)}")
                                # Fall through to standard approach on failure
                        
                        # Standard LLM approach (fallback or primary if AdaLFlow disabled)
                        variant = self._generate_variant_with_llm(
                            base_prompt, 
                            variation_instructions, 
                            temperature_boost=temp_boost
                        )
                        
                        if variant:
                            variants.append(variant)
                            prompt_variants += 1
                            variant_count += 1
                            break  # Success, no need for more retries
                        elif retry < max_retries:
                            logger.warning(f"Variant generation attempt {retry+1}/{max_retries+1} failed, retrying...")
                        else:
                            logger.error(f"Failed to generate variant after {max_retries+1} attempts")
                            
                    except Exception as e:
                        logger.error(f"Error in variant generation (retry {retry+1}/{max_retries+1}): {str(e)}")
                        if retry == max_retries:
                            logger.error("Exceeded maximum retries, skipping this variant")
                    
                    # Small delay between retries to avoid overwhelming the API
                    if retry < max_retries:
                        time.sleep(0.5)
        
        logger.info(f"Successfully generated {len(variants)} variants")
        return variants
    
    def _generate_variant_with_llm(self, base_prompt, variation_instructions="", temperature_boost=0.0, timeout=30):
        """
        Generate a variant of a prompt using an LLM with timeout protection.
        
        Args:
            base_prompt: The Prompt object to create a variation of.
            variation_instructions: Additional instructions for variation.
            temperature_boost: Amount to increase temperature by for more variation.
            timeout: Maximum time in seconds to wait for LLM response.
            
        Returns:
            A new Prompt object or None if generation failed.
        """
        logger = logging.getLogger(__name__)
        try:
            variation_prompt = (
                f"Create a variation of this prompt that preserves its intent "
                f"but changes its wording and structure. {variation_instructions}\n\n"
                f"Original prompt: {base_prompt.content}\n\n"
                f"Variation:"
            )
            
            messages = [{"role": "user", "content": variation_prompt}]
            
            # Use concurrent.futures to enforce a timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a future for the LLM call
                future = executor.submit(
                    self.llm_call,
                    messages=messages,
                    model=self.model_name,
                    temperature=min(0.9, self.temperature + temperature_boost),
                    openrouter_api_key=self.openrouter_api_key
                )
                
                try:
                    # Wait for the future to complete with a timeout
                    response = future.result(timeout=timeout)
                    
                    # Extract the variant content
                    variant_content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    if variant_content:
                        logger.debug(f"Successfully generated variant (length: {len(variant_content)})")
                        variant = Prompt(
                            content=variant_content,
                            score=0.0,
                            feedback_or_generation="Generated variant",
                            generation=base_prompt.generation + 1
                        )
                        return variant
                    logger.warning("Empty variant content returned from LLM")
                    return None
                    
                except concurrent.futures.TimeoutError:
                    logger.error(f"Timeout ({timeout}s) exceeded while generating variant")
                    future.cancel()  # Attempt to cancel the future
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating variant: {str(e)}")
            return None
    
    def _wrap_prompts_as_params(self, prompt_list):
        """
        Convert list of Prompt objects into AdaLFlow Parameters.
        
        Args:
            prompt_list (List[Prompt]): List of Prompt objects to convert.
            
        Returns:
            List[Parameter]: List of AdaLFlow Parameter objects.
        """
        parameters = []
        for idx, prompt in enumerate(prompt_list):
            param = Parameter(
                data=prompt.content,
                role_desc=f"Prompt #{idx}",
                requires_opt=True,
                param_type=ParameterType.PROMPT,
                score=prompt.score or 0.0
            )
            parameters.append(param)
        return parameters
    
    def optimize(self, prompt_objs, feedback_map, score_map, database=None):
        """
        Optimize a batch of prompts based on evaluation feedback.
        
        This method orchestrates the optimization process based on whether 
        AdaLFlow is enabled. It handles the core backward pass functionality.
        
        Args:
            prompt_objs: List of Prompt objects to optimize.
            feedback_map: Dictionary mapping prompt IDs to feedback strings.
            score_map: Dictionary mapping prompt IDs to scores.
            database: Optional database connection to ensure prompts are stored.
            
        Returns:
            List[Prompt]: The optimized prompt objects, or the original prompts if optimization fails.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting backward pass optimization for {len(prompt_objs)} prompts")
        logger.debug(f"Feedback map: {feedback_map}")
        logger.debug(f"Score map: {score_map}")
        logger.info(f"AdaLFlow enabled: {self.adalflow_enabled}")
        
        # If no prompts, return empty list
        if not prompt_objs:
            logger.warning("No prompts provided for optimization")
            return []
        
        # Ensure all prompts are in the database before optimization
        prompt_db_ids = {}
        if database and hasattr(database, 'store_generator_prompt'):
            logger.info("Checking if prompts exist in database before batch optimization")
            for i, prompt in enumerate(prompt_objs):
                # Store prompt in database if not already there
                db_prompt_id = database.store_generator_prompt(
                    content=prompt.content,
                    version=prompt.generation
                )
                if db_prompt_id:
                    prompt_db_ids[str(i)] = db_prompt_id
                    logger.info(f"Prompt {i} stored/retrieved from database with ID: {db_prompt_id}")
                else:
                    logger.warning(f"Failed to store/retrieve prompt {i} in database")
        
        # If AdaLFlow is enabled, use it for optimization
        if self.adalflow_enabled and ADALFLOW_AVAILABLE:
            logger.info("Using AdaLFlow for optimization")
            try:
                # Step 1: Wrap Prompts as AdaLFlow Parameters
                logger.info("Step 1: Wrapping prompts as AdaLFlow parameters")
                parameters = self._wrap_prompts_as_params(prompt_objs)
                logger.debug(f"Created {len(parameters)} AdaLFlow parameters")
                
                # Step 2: Attach gradients (feedback and score)
                logger.info("Step 2: Attaching gradients to parameters")
                for i, param in enumerate(parameters):
                    pid = str(i)
                    feedback = feedback_map.get(pid, "")
                    score = score_map.get(pid, 0.0)
                    magnitude = self._calculate_gradient_magnitude(score)
                    
                    logger.debug(f"Parameter {i}: score={score}, magnitude={magnitude}")
                    
                    param.add_gradient({
                        "score": score,
                        "feedback": feedback,
                        "magnitude": magnitude,
                        "variation_traits": self.variation_traits
                    })
                
                # Step 3: Run Backward Pass
                logger.info("Step 3: Running backward pass")
                for param in parameters:
                    try:
                        logger.debug(f"Running backward() on parameter: {param.role_desc}")
                        param.backward()  # Triggers LLM to analyze feedback
                        logger.debug(f"Backward pass successful for parameter: {param.role_desc}")
                    except Exception as e:
                        logger.error(f"Error in backward() for parameter {param.role_desc}: {str(e)}")
                        # Continue with other parameters even if one fails
                
                # Step 4: Run Optimizer (TGD) to generate new prompt variants
                logger.info("Step 4: Running optimizer")
                try:
                    logger.debug("Setting parameters in optimizer")
                    self.optimizer.set_parameters(parameters)
                    logger.debug("Calling propose() on optimizer")
                    self.optimizer.propose()  # Uses gradients to generate new data
                    logger.debug("Calling step() on optimizer")
                    self.optimizer.step()     # Finalize new values
                    logger.info("Optimizer completed successfully")
                except Exception as e:
                    logger.error(f"Error in optimizer: {str(e)}")
                    logger.warning("Falling back to standard optimization")
                    return self._fallback_optimize(prompt_objs, feedback_map, score_map, database=database)
                
                # Step 5: Return updated prompt objects (new generation)
                logger.info("Step 5: Creating new prompt objects")
                new_prompts = []
                for i, param in enumerate(parameters):
                    original_prompt = prompt_objs[i]
                    logger.debug(f"Creating new prompt from parameter {i}")
                    logger.debug(f"Original content: {original_prompt.content[:50]}...")
                    logger.debug(f"New content: {param.data[:50]}...")
                    
                    prompt = Prompt(
                        content=param.data,
                        score=score_map.get(str(i), 0.0),
                        feedback_or_generation=feedback_map.get(str(i), ""),
                        generation=original_prompt.generation + 1
                    )
                    new_prompts.append(prompt)
                
                # Step 6: Generate additional variants with natural variation
                logger.info("Step 6: Generating additional variants")
                try:
                    variants = self.generate_prompt_variants(
                        new_prompts,
                        variation_traits=self.variation_traits,
                        num_variants=2  # Create 2 additional variants per optimized prompt
                    )
                    logger.info(f"Generated {len(variants)} additional variants")
                    new_prompts.extend(variants)
                    
                    # Save all prompts to database if available
                    if database and hasattr(database, 'store_generator_prompt'):
                        for i, prompt in enumerate(new_prompts):
                            # Determine parent ID from the mapping
                            original_idx = i % len(prompt_objs)  # Map back to original prompt index
                            parent_id = prompt_db_ids.get(str(original_idx))
                            
                            # Store in database
                            if i < len(prompt_objs):  # This is a direct optimization
                                self.save_variants_to_database([prompt], database, parent_prompt_id=parent_id)
                            else:  # This is a variant
                                parent_idx = (i - len(prompt_objs)) % len(prompt_objs)
                                parent_id = prompt_db_ids.get(str(parent_idx))
                                self.save_variants_to_database([prompt], database, parent_prompt_id=parent_id)
                    
                    # The variation is complete - stop the optimization process here
                    logger.info("Optimization and variation completed - stopping optimization process")
                except Exception as e:
                    logger.error(f"Error generating variants: {str(e)}")
                
                return new_prompts
                
            except Exception as e:
                logger.error(f"Error in AdaLFlow optimization: {str(e)}")
                logger.warning("Falling back to standard optimization")
                # Fallback to standard optimization if AdaLFlow fails
                return self._fallback_optimize(prompt_objs, feedback_map, score_map, database=database)
        else:
            # Use standard optimization approach
            logger.info("Using standard optimization (non-AdaLFlow)")
            return self._fallback_optimize(prompt_objs, feedback_map, score_map, database=database)

    def _fallback_optimize(self, prompt_objs, feedback_map, score_map, variant_limit=4, database=None):
        """
        Fallback optimization method using standard LLM approach with safety limits.
        
        Args:
            prompt_objs: List of Prompt objects to optimize.
            feedback_map: Dictionary mapping prompt IDs to feedback strings.
            score_map: Dictionary mapping prompt IDs to scores.
            variant_limit: Maximum number of variants to generate.
            database: Optional database connection to save variants
            
        Returns:
            List[Prompt]: The optimized prompt objects.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Using fallback optimization for {len(prompt_objs)} prompts")
        
        # Ensure all prompts are in the database before optimization
        prompt_db_ids = {}
        if database and hasattr(database, 'store_generator_prompt'):
            logger.info("Checking if prompts exist in database before fallback optimization")
            for i, prompt in enumerate(prompt_objs):
                # Store prompt in database if not already there
                db_prompt_id = database.store_generator_prompt(
                    content=prompt.content,
                    version=prompt.generation
                )
                if db_prompt_id:
                    prompt_db_ids[str(i)] = db_prompt_id
                    logger.info(f"Prompt {i} stored/retrieved from database with ID: {db_prompt_id}")
                else:
                    logger.warning(f"Failed to store/retrieve prompt {i} in database")
        
        # Safety check - don't optimize too many prompts at once
        if len(prompt_objs) > 10:
            logger.warning(f"Large number of prompts ({len(prompt_objs)}) to optimize, limiting to first 10")
            prompt_objs = prompt_objs[:10]
        
        new_prompts = []
        successful_optimizations = 0
        optimization_errors = 0
        
        for i, prompt in enumerate(prompt_objs):
            try:
                pid = str(i)
                feedback = feedback_map.get(pid, "")
                score = score_map.get(pid, 0.0)
                
                logger.debug(f"Optimizing prompt {i}: score={score}")
                
                # Optimize the prompt with timeout protection
                start_time = time.time()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.optimize_prompt,
                        prompt, feedback, score
                    )
                    
                    try:
                        # Wait for the future to complete with a timeout
                        improved_content = future.result(timeout=60)
                        execution_time = time.time() - start_time
                        logger.debug(f"Optimization completed in {execution_time:.2f} seconds")
                        
                    except concurrent.futures.TimeoutError:
                        logger.error("Timeout exceeded while optimizing prompt, using original content")
                        future.cancel()
                        improved_content = prompt.content
                        optimization_errors += 1
                
                logger.debug(f"Original content: {prompt.content[:50]}...")
                logger.debug(f"Improved content: {improved_content[:50]}...")
                
                # Create a new prompt with the improved content
                new_prompt = Prompt(
                    content=improved_content,
                    score=score,
                    feedback_or_generation=feedback,
                    generation=prompt.generation + 1
                )
                new_prompts.append(new_prompt)
                successful_optimizations += 1
                
                # Save the optimized prompt to database
                if database and hasattr(database, 'store_generator_prompt'):
                    parent_id = prompt_db_ids.get(pid)
                    if parent_id:
                        prompt_id = database.store_generator_prompt(
                            content=improved_content,
                            parent_prompt_id=parent_id,
                            version=prompt.generation + 1
                        )
                        logger.info(f"Saved optimized prompt {i} with ID: {prompt_id}")
                
            except Exception as e:
                logger.error(f"Error optimizing prompt {i}: {str(e)}")
                optimization_errors += 1
                
                # Use the original prompt as a fallback
                if prompt not in new_prompts:
                    new_prompts.append(prompt)
        
        # Determine safe variant count based on success rate
        if successful_optimizations == 0:
            logger.error("All prompt optimizations failed, skipping variant generation")
            return new_prompts
        
        # Calculate safe variant count - reduce if we had errors
        if optimization_errors > 0:
            safe_variant_count = max(1, min(2, variant_limit - optimization_errors))
            logger.warning(f"Reducing variant count to {safe_variant_count} due to {optimization_errors} optimization errors")
        else:
            safe_variant_count = min(variant_limit, 2)
        
        # Generate variants with a timeout
        logger.info(f"Generating variants (max {safe_variant_count}) for fallback optimization")
        try:
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.generate_prompt_variants,
                    new_prompts,
                    variation_traits=self.variation_traits,
                    num_variants=safe_variant_count
                )
                
                try:
                    # Wait for the future to complete with a timeout
                    variants = future.result(timeout=120)  # 2 minute timeout for the entire variant generation
                    generation_time = time.time() - start_time
                    logger.info(f"Generated {len(variants)} variants in {generation_time:.2f} seconds")
                    
                    # Save variants to database
                    if database and hasattr(database, 'store_generator_prompt'):
                        for j, new_prompt in enumerate(new_prompts):
                            # Find original prompt index
                            orig_idx = j % len(prompt_objs)
                            parent_db_id = prompt_db_ids.get(str(orig_idx))
                            
                            # Get variants related to this prompt
                            prompt_variants = []
                            for variant in variants:
                                # Simple heuristic to match variants to their base prompts
                                # This is a simplification; in reality, need a better method to track this relationship
                                if variant.generation == new_prompt.generation + 1:
                                    prompt_variants.append(variant)
                            
                            # Save them to database
                            if prompt_variants and parent_db_id:
                                self.save_variants_to_database(prompt_variants, database, parent_prompt_id=parent_db_id)
                    
                    # Add variants to new_prompts
                    new_prompts.extend(variants)
                    
                    # The variation is complete - stop the optimization process here
                    logger.info("Optimization and variation completed - stopping optimization process")
                except concurrent.futures.TimeoutError:
                    logger.error("Timeout exceeded while generating variants, proceeding with optimized prompts only")
                    future.cancel()
            
        except Exception as e:
            logger.error(f"Error generating variants in fallback: {str(e)}")
        
        # Final safety check - make sure we return something
        if not new_prompts and prompt_objs:
            logger.warning("Fallback optimization produced no results, returning original prompts")
            return prompt_objs
        
        return new_prompts
        
    def save_variants_to_database(self, variants, database, parent_prompt_id=None):
        """
        Save generated prompt variants to the database.
        
        Args:
            variants: List of Prompt objects to save
            database: Database instance with store_generator_prompt method
            parent_prompt_id: Optional ID of the parent prompt
            
        Returns:
            List of IDs of the saved variants
        """
        if not database or not hasattr(database, 'store_generator_prompt'):
            logger.warning("No valid database connection for saving variants")
            return []
            
        logger.info(f"Saving {len(variants)} variants to database")
        saved_ids = []
        
        for i, variant in enumerate(variants):
            try:
                prompt_id = database.store_generator_prompt(
                    content=variant.content,
                    parent_prompt_id=parent_prompt_id,
                    version=variant.generation
                )
                
                if prompt_id:
                    logger.info(f"Saved variant {i+1} with ID: {prompt_id}")
                    saved_ids.append(prompt_id)
                    
                    # Update performance metrics
                    success = database.update_generator_prompt_performance(
                        prompt_id=prompt_id,
                        avg_score=variant.score or 0.0,
                        rounds_survived=1,  # New variant
                        is_active=True
                    )
                    
                    if not success:
                        logger.warning(f"Failed to update performance for variant {i+1}")
            except Exception as e:
                logger.error(f"Error saving variant {i+1}: {str(e)}")
                
        return saved_ids

    def optimize_model(self, model, feedback_map, score_map):
        """
        Optimize prompts in a Model instance using AdaLFlow.
        
        This updates the model's prompts directly based on feedback.
        
        Args:
            model: The Model instance to optimize.
            feedback_map: Dictionary mapping prompt IDs to feedback strings.
            score_map: Dictionary mapping prompt IDs to scores.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting model optimization for {model.name}")
        logger.debug(f"Feedback map: {feedback_map}")
        logger.debug(f"Score map: {score_map}")
        logger.info(f"AdaLFlow enabled: {self.adalflow_enabled}")
        
        if not model.adalflow_enabled:
            logger.warning(f"Model {model.name} does not have AdaLFlow enabled")
            return False
            
        if not hasattr(model, 'adalflow_params') or not model.adalflow_params:
            logger.warning(f"Model {model.name} has no AdaLFlow parameters")
            return False
            
        logger.info(f"Model has {len(model.adalflow_params)} AdaLFlow parameters")
        
        try:
            # Step 1: Attach gradients to model parameters with magnitude information
            logger.info("Step 1: Attaching gradients to model parameters")
            for prompt_id, param in model.adalflow_params.items():
                feedback = feedback_map.get(prompt_id, "")
                score = score_map.get(prompt_id, 0.0)
                magnitude = self._calculate_gradient_magnitude(score)
                
                logger.debug(f"Parameter {prompt_id}: score={score}, magnitude={magnitude}")
                
                param.add_gradient({
                    "score": score,
                    "feedback": feedback,
                    "magnitude": magnitude,
                    "variation_traits": self.variation_traits
                })
            
            # Step 2: Run backward pass
            logger.info("Step 2: Running backward pass on model parameters")
            for prompt_id, param in model.adalflow_params.items():
                try:
                    logger.debug(f"Running backward() on parameter {prompt_id}")
                    param.backward()
                    logger.debug(f"Backward pass successful for parameter {prompt_id}")
                except Exception as e:
                    logger.error(f"Error in backward() for parameter {prompt_id}: {str(e)}")
                    # Continue with other parameters even if one fails
            
            # Step 3: Run optimizer
            logger.info("Step 3: Running optimizer on model parameters")
            try:
                logger.debug("Setting parameters in optimizer")
                self.optimizer.set_parameters(list(model.adalflow_params.values()))
                logger.debug("Calling propose() on optimizer")
                self.optimizer.propose()
                logger.debug("Calling step() on optimizer")
                self.optimizer.step()
                logger.info("Optimizer completed successfully")
            except Exception as e:
                logger.error(f"Error in optimizer for model: {str(e)}")
                return False
            
            # Step 4: Update the model prompts
            logger.info("Step 4: Updating model prompts")
            updates_count = 0
            for prompt_id, param in model.adalflow_params.items():
                if prompt_id in model.prompts:
                    try:
                        old_content = model.prompts[prompt_id].content
                        new_content = param.data
                        score = score_map.get(prompt_id, 0.0)
                        feedback = feedback_map.get(prompt_id, "")
                        
                        logger.debug(f"Updating prompt {prompt_id}")
                        logger.debug(f"Old content: {old_content[:50]}...")
                        logger.debug(f"New content: {new_content[:50]}...")
                        
                        model.update_prompt(
                            prompt_id=prompt_id,
                            new_content=new_content,
                            score=score,
                            feedback=feedback
                        )
                        updates_count += 1
                    except Exception as e:
                        logger.error(f"Error updating prompt {prompt_id}: {str(e)}")
            
            # The optimization is complete - stop the process here
            logger.info(f"Updated {updates_count} prompts in the model")
            logger.info("Model optimization completed - stopping optimization process")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            return False

    def optimize_model_with_evaluation(self, model, feedback_map, score_map, input_data, prompt_evaluator, database=None):
        """
        Optimize prompts in a Model instance using the integrated prompt structure.
        
        Args:
            model: The Model object containing prompts to optimize.
            feedback_map: prompt_id -> str feedback mapping.
            score_map: prompt_id -> float score mapping.
            input_data: Data to use for evaluating new prompts.
            prompt_evaluator: PromptEvaluator instance for evaluating new prompts.
            database: Optional database connection to save variants
            
        Returns:
            List of best performing Prompt objects after optimization and evaluation.
        """
        logger = logging.getLogger(__name__)
        
        # Step 0: Ensure all prompts being optimized are in the database
        if database and hasattr(database, 'store_generator_prompt'):
            logger.info("Checking if prompts exist in database before optimization")
            prompt_id_map = {}
            for prompt_id, prompt in model.prompts.items():
                # Store prompt in database if not already there
                db_prompt_id = database.store_generator_prompt(
                    content=prompt.content,
                    version=prompt.generation
                )
                if db_prompt_id:
                    prompt_id_map[prompt_id] = db_prompt_id
                    logger.info(f"Prompt {prompt_id} stored/retrieved from database with ID: {db_prompt_id}")
                else:
                    logger.warning(f"Failed to store/retrieve prompt {prompt_id} in database")
        
        # Step 1: Optimize the model's prompts directly
        if model.adalflow_enabled and self.adalflow_enabled:
            optimized = self.optimize_model(model, feedback_map, score_map)
            if not optimized:
                # If AdaLFlow optimization failed, use standard approach
                variant_prompts = []
                for prompt_id, prompt in model.prompts.items():
                    improved_content = self.optimize_prompt(
                        prompt, 
                        feedback_map.get(prompt_id, ""),
                        score_map.get(prompt_id, 0.0)
                    )
                    model.update_prompt(
                        prompt_id=prompt_id,
                        new_content=improved_content,
                        score=score_map.get(prompt_id, 0.0),
                        feedback=feedback_map.get(prompt_id, "")
                    )
        
        # Step 2: Generate additional prompt variants
        variant_prompts = []
        for prompt_id, prompt in model.prompts.items():
            variants = self.generate_prompt_variants(
                [prompt], 
                variation_traits=self.variation_traits, 
                num_variants=3
            )
            variant_prompts.extend(variants)
            
            # Save variants to database if available
            if database:
                # Use the mapped database ID as parent if available
                parent_id = prompt_id_map.get(prompt_id, prompt_id) if database and prompt_id_map else prompt_id
                self.save_variants_to_database(variants, database, parent_prompt_id=parent_id)
        
        # Step 3: Add all prompts to a combined list for evaluation
        all_prompts = list(model.prompts.values()) + variant_prompts
        
        # Step 4: Evaluate all prompts using the promptEvaluator
        if prompt_evaluator:
            best_prompts = prompt_evaluator.evaluate_and_select_best(
                prompts=all_prompts,
                input_data=input_data
            )
            return best_prompts
        else:
            # If no evaluator is provided, return all prompts
            return all_prompts 