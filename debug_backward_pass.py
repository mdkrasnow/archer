#!/usr/bin/env python
"""
Debug script specifically for the backward pass.

This script directly tests the backward pass functionality with
detailed inspection of each step to help identify why it's failing.
"""

import os
import sys
import logging
import inspect
from pathlib import Path
import json
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backward_pass_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import components
from data_labelling.archer.archer import Archer
from data_labelling.archer.helpers.prompt import Prompt
from data_labelling.archer.backwardPass.danielson_model import DanielsonModel
from data_labelling.archer.backwardPass.promptOptimizer import PromptOptimizer, ADALFLOW_AVAILABLE


def inspect_object(obj, name="object", max_depth=1, current_depth=0):
    """
    Recursively inspect an object and log its attributes.
    
    Args:
        obj: The object to inspect
        name: Name to use in logging
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """
    indent = "  " * current_depth
    
    if current_depth > max_depth:
        logger.debug(f"{indent}{name}: [max depth reached]")
        return
        
    # Log the object type
    obj_type = type(obj).__name__
    logger.debug(f"{indent}{name} (type: {obj_type}):")
    
    # For basic types, just log the value
    if isinstance(obj, (str, int, float, bool, type(None))):
        logger.debug(f"{indent}  value: {obj}")
        return
        
    # For lists, log length and first few items
    if isinstance(obj, list):
        logger.debug(f"{indent}  length: {len(obj)}")
        if len(obj) > 0:
            for i, item in enumerate(obj[:3]):
                if current_depth < max_depth:
                    inspect_object(item, f"{name}[{i}]", max_depth, current_depth + 1)
            if len(obj) > 3:
                logger.debug(f"{indent}  ... ({len(obj) - 3} more items)")
        return
        
    # For dictionaries, log keys and first few values
    if isinstance(obj, dict):
        logger.debug(f"{indent}  keys: {list(obj.keys())}")
        if len(obj) > 0:
            for k, v in list(obj.items())[:3]:
                if current_depth < max_depth:
                    inspect_object(v, f"{name}['{k}']", max_depth, current_depth + 1)
            if len(obj) > 3:
                logger.debug(f"{indent}  ... ({len(obj) - 3} more items)")
        return
    
    # For other objects, log attributes
    try:
        attrs = dir(obj)
        attrs = [a for a in attrs if not a.startswith('__')]
        logger.debug(f"{indent}  attributes: {attrs[:10]}" + 
                    (f" ... ({len(attrs) - 10} more)" if len(attrs) > 10 else ""))
        
        # Log a few important attributes for known types
        if obj_type == 'Prompt':
            logger.debug(f"{indent}  content: {obj.content[:50]}...")
            logger.debug(f"{indent}  score: {getattr(obj, 'score', 'N/A')}")
            logger.debug(f"{indent}  generation: {getattr(obj, 'generation', 'N/A')}")
        
        elif obj_type == 'Parameter' and hasattr(obj, 'data'):
            logger.debug(f"{indent}  data: {obj.data[:50]}...")
            logger.debug(f"{indent}  role_desc: {getattr(obj, 'role_desc', 'N/A')}")
        
        elif obj_type == 'Archer':
            logger.debug(f"{indent}  adalflow_enabled: {getattr(obj, 'adalflow_enabled', 'N/A')}")
            if hasattr(obj, 'active_prompts'):
                logger.debug(f"{indent}  active_prompts: {len(obj.active_prompts)}")
    except Exception as e:
        logger.debug(f"{indent}  error inspecting: {str(e)}")


def debug_adalflow_status():
    """Debug the AdaLFlow availability and status."""
    logger.info("=== CHECKING ADALFLOW STATUS ===")
    
    logger.info(f"ADALFLOW_AVAILABLE: {ADALFLOW_AVAILABLE}")
    
    # Try to import AdaLFlow components directly
    try:
        logger.info("Attempting to import AdaLFlow components directly")
        
        try:
            from adalflow.optim.parameter import Parameter, ParameterType
            logger.info("  ✓ Successfully imported Parameter and ParameterType")
        except ImportError as e:
            logger.error(f"  ✗ Failed to import Parameter: {str(e)}")
        
        try:
            from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer
            logger.info("  ✓ Successfully imported TGDOptimizer")
        except ImportError as e:
            logger.error(f"  ✗ Failed to import TGDOptimizer: {str(e)}")
        
        try:
            from adalflow.core import Generator
            logger.info("  ✓ Successfully imported Generator")
        except ImportError as e:
            logger.error(f"  ✗ Failed to import Generator: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error during AdaLFlow imports: {str(e)}")
    
    # Check if we're using the mock classes
    if not ADALFLOW_AVAILABLE:
        logger.info("Using mock AdaLFlow classes")
        
        optimizer = PromptOptimizer(
            model_name="test-model",
            adalflow_enabled=True
        )
        
        logger.info(f"Mock optimizer.optimizer type: {type(optimizer.optimizer).__name__}")
        logger.info(f"Mock optimizer.generator type: {type(optimizer.generator).__name__}")
    else:
        logger.info("Using real AdaLFlow classes")


def debug_backward_pass_steps():
    """Debug each step of the backward pass in isolation."""
    logger.info("=== DEBUGGING BACKWARD PASS STEPS ===")
    
    # Create test objects
    initial_prompts = [
        Prompt("Test prompt 1: Generate an analysis for component {component_id} from {input}"),
        Prompt("Test prompt 2: Create an evaluation for component {component_id} using {input}")
    ]
    
    # Create test evaluations
    test_evaluations = [
        (
            initial_prompts[0],
            "Generated content for prompt 1",
            {"score": 3.5, "feedback": "Test feedback for prompt 1"}
        ),
        (
            initial_prompts[1],
            "Generated content for prompt 2",
            {"score": 4.0, "feedback": "Test feedback for prompt 2"}
        )
    ]
    
    # Create optimizer with logging for each step
    optimizer = PromptOptimizer(
        model_name="test-model",
        temperature=0.7,
        adalflow_enabled=True,  # Test with AdaLFlow enabled
        openrouter_api_key="test_api_key"
    )
    
    # Log the optimizer status
    logger.info("Optimizer status:")
    logger.info(f"  adalflow_enabled: {optimizer.adalflow_enabled}")
    logger.info(f"  ADALFLOW_AVAILABLE: {ADALFLOW_AVAILABLE}")
    
    # Debug the _wrap_prompts_as_params method
    logger.info("Testing _wrap_prompts_as_params method")
    try:
        parameters = optimizer._wrap_prompts_as_params(initial_prompts)
        logger.info(f"  ✓ Created {len(parameters)} parameters")
        for i, param in enumerate(parameters):
            logger.info(f"    Parameter {i}: {param.data[:30]}...")
    except Exception as e:
        logger.error(f"  ✗ Error in _wrap_prompts_as_params: {str(e)}")
        traceback.print_exc()
    
    # Debug parameters and optimization
    logger.info("Creating parameters and testing backward")
    try:
        # Create feedback and score maps
        feedback_map = {"0": "Test feedback 1", "1": "Test feedback 2"}
        score_map = {"0": 3.5, "1": 4.0}
        
        if optimizer.adalflow_enabled and ADALFLOW_AVAILABLE:
            # Define a mock backward method for debugging
            def mock_backward(self):
                logger.info(f"Mock backward called for {self.role_desc}")
                # Here you can add additional checks
            
            # Replace the actual backward method with our mock
            from adalflow.optim.parameter import Parameter
            original_backward = Parameter.backward
            Parameter.backward = mock_backward
            
            # Run the optimization
            try:
                logger.info("Running optimization with mock backward")
                new_prompts = optimizer.optimize(initial_prompts, feedback_map, score_map)
                logger.info(f"  ✓ Optimization completed, got {len(new_prompts)} new prompts")
                
                # Inspect the results
                for i, prompt in enumerate(new_prompts):
                    logger.info(f"    Prompt {i}: {prompt.content[:50]}...")
            finally:
                # Restore original method
                Parameter.backward = original_backward
        else:
            logger.info("Running fallback optimization (AdaLFlow not available)")
            new_prompts = optimizer.optimize(initial_prompts, feedback_map, score_map)
            logger.info(f"  ✓ Fallback optimization completed, got {len(new_prompts)} new prompts")
    except Exception as e:
        logger.error(f"  ✗ Error in optimization: {str(e)}")
        traceback.print_exc()
    
    # Now debug with a real Archer object
    logger.info("Testing backward pass with Archer")
    try:
        # Create an Archer instance with AdaLFlow enabled
        archer = Archer(
            generator_model_name="test-model",
            evaluator_model_name="test-model",
            optimizer_model_name="test-model",
            knowledge_base=["./data_labelling/eval"],
            rubric="Test rubric",
            initial_prompts=initial_prompts,
            openrouter_api_key="test_api_key",
            adalflow_enabled=True
        )
        
        # Replace optimizer with our debug version
        archer.optimizer = optimizer
        
        # Set the active prompts
        archer.active_prompts = initial_prompts
        
        # Log Archer status
        logger.info(f"Archer adalflow_enabled: {archer.adalflow_enabled}")
        logger.info(f"Archer optimizer.adalflow_enabled: {archer.optimizer.adalflow_enabled}")
        
        # Run the backward pass
        logger.info("Running Archer backward pass")
        archer.run_backward_pass(test_evaluations)
        
        # Check if the active prompts were updated
        logger.info(f"Active prompts after backward pass: {len(archer.active_prompts)}")
        for i, prompt in enumerate(archer.active_prompts):
            logger.info(f"  Prompt {i}: {prompt.content[:50]}...")
    except Exception as e:
        logger.error(f"Error in Archer backward pass: {str(e)}")
        traceback.print_exc()


def debug_danielson_backward_pass():
    """Debug the backward pass with the Danielson model."""
    logger.info("=== DEBUGGING DANIELSON MODEL BACKWARD PASS ===")
    
    # Create the Danielson model
    model = DanielsonModel(adalflow_enabled=True)
    
    # Log model status
    logger.info(f"Model name: {model.name}")
    logger.info(f"Model adalflow_enabled: {model.adalflow_enabled}")
    logger.info(f"Model has {len(model.prompts)} prompts")
    logger.info(f"Model has {len(model.adalflow_params) if hasattr(model, 'adalflow_params') else 0} adalflow_params")
    
    # Create feedback and score maps
    feedback_map = {
        "context_analysis": "Could be more specific",
        "component_evaluation_base": "Needs better structure"
    }
    
    score_map = {
        "context_analysis": 3.5,
        "component_evaluation_base": 4.0
    }
    
    # Create optimizer
    optimizer = PromptOptimizer(
        model_name="test-model",
        temperature=0.7,
        adalflow_enabled=True,
        openrouter_api_key="test_api_key"
    )
    
    # Debug optimize_model
    logger.info("Testing optimize_model with Danielson model")
    try:
        result = optimizer.optimize_model(model, feedback_map, score_map)
        logger.info(f"optimize_model result: {result}")
    except Exception as e:
        logger.error(f"Error in optimize_model: {str(e)}")
        traceback.print_exc()


def main():
    """Run all debug functions."""
    logger.info("Starting backward pass debugging")
    
    try:
        # Check AdaLFlow status
        debug_adalflow_status()
        
        # Debug backward pass steps
        debug_backward_pass_steps()
        
        # Debug Danielson model backward pass
        debug_danielson_backward_pass()
        
        logger.info("Debugging completed")
    except Exception as e:
        logger.error(f"Unhandled exception in debugging: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 