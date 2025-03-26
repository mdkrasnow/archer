# Archer - Prompt Optimization System

Archer is a system for optimizing prompts through iterative feedback and evaluation.

## System Components

- **Forward Pass**: Generates content using current prompts and evaluates it
  - `GenerativeModel`: Generates content using LLMs
  - `AIExpert`: Evaluates generated content against a rubric
  
- **Backward Pass**: Optimizes prompts based on evaluation feedback
  - `PromptOptimizer`: Improves prompts using evaluation feedback
  
- **Helpers**: Utility functions and classes
  - `Prompt`: Represents a prompt with content, score, and feedback
  - `PerformanceTracker`: Tracks and visualizes prompt performance over generations
  - `llm_call`: Function for making API calls to language models

## Running Tests

To run the tests, you'll need pytest installed:

```bash
pip install pytest
```

Then, from the `data-labelling/archer` directory, run:

```bash
pytest
```

Or to run specific test files:

```bash
# Run all tests for Archer
pytest data-labelling/archer/tests/test_archer.py

# Run tests for LLM call function
pytest tests/helpers/test_llm_call.py
```

## Test Structure

- `tests/test_archer.py`: Tests for the main Archer class
- `tests/helpers/test_llm_call.py`: Tests for the LLM call function 