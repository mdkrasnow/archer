# Data Labelling with Archer

## Introduction

This directory contains the Archer system, a sophisticated prompt optimization framework that uses iterative feedback and evaluation to continuously improve LLM prompts. Archer orchestrates a feedback loop of generation, evaluation, and optimization to create high-quality prompts for specific tasks.

## Archer Architecture

Archer has three main components:

1. **Forward Pass**: Generates content and evaluates it against a rubric
   - `GenerativeModel`: Generates content using LLMs
   - `AIExpert`: Evaluates generated content using evaluation criteria

2. **Backward Pass**: Optimizes prompts based on evaluation feedback
   - `PromptOptimizer`: Improves prompts using evaluation feedback and performance scores

3. **Helpers**: Utility functions and classes
   - `Prompt`: Represents a prompt with content, score, and feedback
   - `PerformanceTracker`: Tracks and visualizes prompt performance over generations
   - `llm_call`: Function for making API calls to language models

## Step-by-Step Guide to Implementing Archer

### 1. Installation Requirements

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Preparing Your Knowledge Base

Create directories containing text files with relevant domain knowledge for your task. These will be used by the AIExpert for context-aware evaluations.

### 3. Creating Your Evaluation Rubric

Create a string that contains the evaluation criteria. For example:

```python
rubric = """
Evaluate the generated content on the following criteria:
1. Accuracy (1-5): Is the information factually correct?
2. Completeness (1-5): Does it address all aspects of the input?
3. Clarity (1-5): Is the content clear and well-structured?
4. Relevance (1-5): Is the content relevant to the input?
5. Helpfulness (1-5): Will this content help the user achieve their goal?
"""
```

### 4. Creating Initial Prompts

Create a list of initial prompts to start the optimization process:

```python
from data_labelling.archer.helpers.prompt import Prompt

initial_prompts = [
    Prompt("You are a helpful assistant. Please respond to the following query with accurate information: {input}"),
    Prompt("As an expert in the field, please provide a comprehensive answer to: {input}")
]
```

### 5. Instantiating an Archer Model

Here's how to initialize an Archer instance:

```python
from data_labelling.archer.archer import Archer

# Directories containing your domain knowledge
knowledge_base_dirs = ["./knowledge/domain1", "./knowledge/domain2"]

# Initialize Archer
archer = Archer(
    generator_model_name="gemini-2.0-flash",              # Model for content generation
    evaluator_model_name="gemini-2.0-flash",      # Model for evaluation
    optimizer_model_name="gemini-2.0-flash",    # Model for prompt optimization
    knowledge_base=knowledge_base_dirs,        # List of directory paths
    rubric=rubric,                             # Evaluation criteria
    initial_prompts=initial_prompts,           # Starting prompts
    openrouter_api_key="your_api_key_here",    # API key for LLM access
    # Optional parameters
    input_spec="string",                       # Input data type
    output_spec="string",                      # Output data type
    human_validation_enabled=False,            # Whether to use human validation
    num_simulations_per_prompt=3,              # Number of test cases per prompt
    max_prompts_per_cycle=4                    # Max prompts to use in each cycle
)
```

### 6. Running a Training Loop

To optimize prompts through multiple cycles:

```python
# Define a function that provides input data for training
def input_data_generator():
    # You can use random samples from a dataset
    # or predefined examples
    examples = [
        "What is machine learning?",
        "Explain the concept of neural networks.",
        "How do transformers work in NLP?"
    ]
    return random.choice(examples)

# Run the training loop for 5 cycles
archer.run_training_loop(input_data_generator, num_cycles=5)
```

### 7. Using Optimized Prompts

After training, you can access the best prompts:

```python
# Get the best performing prompts
best_prompts = archer._select_top_prompts()

# Print the best prompt
print(f"Best Prompt: {best_prompts[0].content}")
print(f"Score: {best_prompts[0].score}")
```

### 8. Single Training Cycle

If you want to run just one training cycle:

```python
input_data = "Explain the concept of backpropagation in neural networks."
results = archer.run_training_cycle(input_data)

# Results contain evaluations of the generated content
for result in results:
    print(f"Score: {result['score']}")
    print(f"Feedback: {result['feedback']}")
```

## Advanced Configuration

### Enabling Human Validation

```python
archer = Archer(
    # ... other parameters ...
    human_validation_enabled=True
)
```

### Using AdaLflow for Optimization

```python
archer = Archer(
    # ... other parameters ...
    adalflow_enabled=True,
    adalflow_config={
        'temperature': 0.8,
        'max_trials': 10,
        'top_k': 5
    }
)
```

## Example Use Case: Content Generation

This example shows how to set up Archer for optimizing prompts for a content generation task:

```python
# Define rubric for content generation
content_rubric = """
Evaluate the generated content on:
1. Engagement (1-5): Is it interesting and engaging?
2. Style (1-5): Is the writing style appropriate?
3. Structure (1-5): Is it well-organized?
4. Accuracy (1-5): Is the information correct?
5. Completeness (1-5): Does it cover the topic thoroughly?
"""

# Initial prompts for content generation
content_prompts = [
    Prompt("Write an engaging blog post about {input}. Include key facts, examples, and a conclusion."),
    Prompt("Create an informative article about {input} with an introduction, main points, and summary.")
]

# Initialize Archer for content generation
content_archer = Archer(
    generator_model_name="gemini-2.0-flash",
    evaluator_model_name="gemini-2.0-flash",
    optimizer_model_name="gemini-2.0-flash",
    knowledge_base=["./knowledge/writing", "./knowledge/blogging"],
    rubric=content_rubric,
    initial_prompts=content_prompts,
    openrouter_api_key="your_api_key_here"
)

# Run optimization
content_archer.run_training_loop(
    lambda: random.choice(["artificial intelligence", "climate change", "space exploration"]),
    num_cycles=3
)

# Get the optimized prompt
best_content_prompt = content_archer._select_top_prompts()[0]
print(f"Best Content Generation Prompt: {best_content_prompt.content}")
```

## Best Practices

1. **Start with diverse initial prompts** to explore different approaches
2. **Create a detailed rubric** that precisely defines what makes a good output
3. **Use domain-specific knowledge** in your knowledge base
4. **Run multiple training cycles** to allow prompts to converge
5. **Test optimized prompts** on new inputs to ensure generalizability

## Troubleshooting

- If prompts aren't improving, try increasing `num_simulations_per_prompt`
- If all prompts get similar scores, make your rubric more discriminative
- If optimization is slow, reduce `max_prompts_per_cycle`
- Ensure your API key has access to the models you're trying to use 