# Argilla Database Integration for Archer

This module provides a robust integration between the Archer system and Argilla for data labeling and feedback collection.

## Overview

The `ArgillaDatabase` class handles all interactions with the Argilla database, including:

- Storing and retrieving generated content
- Storing and retrieving evaluations
- Managing prompts and their performance metrics
- Tracking prompt evolution across generations

## Usage Examples

### Initialization

```python
from database.argilla import ArgillaDatabase

# Using default environment variables
db = ArgillaDatabase()

# With explicit credentials
db = ArgillaDatabase(
    api_url="http://localhost:6900",
    api_key="admin.apikey"
)
```

### Connecting to Argilla

```python
# Connect to the Argilla server
success = db.connect()
if not success:
    print("Failed to connect to Argilla server")
```

### Storing Generated Content

```python
# Store a piece of generated content
output_id = db.store_generated_content(
    input_data="User query about creating a web app",
    content="Here's how to create a React web app...",
    prompt_id="prompt-123",
    round_num=1
)

if output_id:
    print(f"Successfully stored content with ID: {output_id}")
else:
    print("Failed to store content")
```

### Storing Evaluations

```python
# Store an AI evaluation
success = db.store_evaluation(
    output_id="output-123",
    score=4,
    feedback="Good response but could be more detailed",
    improved_output="Here's an improved version...",
    is_human=False
)

# Store human feedback
success = db.store_human_feedback(
    output_id="output-123",
    score=3,
    feedback="The response is misleading in the second paragraph",
    improved_output="Here's how it should be..."
)
```

### Managing Prompts

```python
# Store a new prompt
prompt_id = db.store_prompt(
    prompt_text="Generate a tutorial for...",
    model="gpt-4",
    purpose="tutorial generation",
    generation=0
)

# Update prompt performance
db.update_prompt_performance(
    prompt_id="prompt-123",
    avg_score=4.2,
    survived=True
)

# Get best performing prompts
top_prompts = db.get_current_best_prompts(top_n=3)
```

### Retrieving Data

```python
# Get data for annotation
annotation_data = db.get_current_data_for_annotation(
    round_num=2,
    limit=10
)

# Get performance metrics
metrics = db.get_performance_metrics(max_rounds=2)

# Get prompt evolution history
prompt_history = db.get_prompt_history()
```

## Error Handling

The ArgillaDatabase class includes robust error handling with detailed logging. All methods include try-except blocks to catch and log exceptions, returning appropriate default values when operations fail.

## Testing

Run the tests with:

```
python -m unittest tests.test_argilla_database
```

## Dependencies

- argilla
- pandas
- numpy
- uuid
- datetime
- logging 