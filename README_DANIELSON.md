# Archer-Danielson Framework Implementation

This directory contains an implementation of the Archer framework for optimizing prompts used to generate Danielson framework component summaries.

## Overview

The Archer-Danielson integration combines:

1. **Archer**: A sophisticated prompt optimization framework that uses iterative feedback and evaluation to continuously improve LLM prompts
2. **Danielson Framework**: A teacher evaluation framework with multiple components across domains like Planning and Preparation, Classroom Environment, Instruction, and Professional Responsibilities

This implementation allows users to generate and evaluate summaries for Danielson framework components based on low-inference notes from classroom observations. The system uses Archer to optimize the prompts that generate these summaries, improving their quality over time through an iterative feedback process.

## Features

- Generate summaries for any Danielson component (1a through 3e)
- Evaluate the quality of generated summaries based on evidence, framework alignment, clarity, actionability, and professionalism
- Provide human feedback and perfect output examples
- Store data in Argilla for tracking and analysis
- Trigger optimization to improve prompt quality over time

## Prerequisites

Before running the application, ensure you have:

1. Python 3.8+ installed
2. Required Python packages (install with `pip install -r requirements.txt`)
3. Environment variables set in a `.env` file:
   - `OPENROUTER_API_KEY`: API key for accessing language models
   - `ARGILLA_API_URL`: URL for the Argilla database (default: http://localhost:6900)
   - `ARGILLA_API_KEY`: API key for Argilla (default: admin.apikey)

## Running the Application

You can run the Archer-Danielson application using the provided script:

```bash
python data_labelling/app.py
```

Optional arguments:
- `--share`: Create a shareable public link for the Gradio interface
- `--port PORT`: Specify the port to run the server on (default: 7860)
- `--no-archer`: Run without the Archer optimization framework (lighter mode)

Example:
```bash
python data_labelling/app.py --port 8080
```

## Using the Interface

1. **Generate Input**: Click the "Generate Random Input" button to create a random pair of low-inference notes and Danielson component
2. **Generate Summary**: Click "Generate Summary" to use the current prompts to create a component summary
3. **Evaluate**: Review the generated summary and adjust the score, feedback, and perfect output example as needed
4. **Save & Get New Data**: Save your evaluation and generate new data
5. **Run Optimization**: Trigger the backward pass in Archer to optimize prompts based on feedback

## How It Works

### Architecture

The implementation consists of several key components:

1. **DanielsonArcherApp**: The main application class that coordinates the Gradio interface
2. **Archer**: The prompt optimization framework with forward and backward passes
3. **DanielsonModel**: Adapter for the Danielson framework to work with Archer
4. **ArgillaDatabase**: Database integration for storing data and tracking performance

### Process Flow

1. The user selects or generates low-inference notes and a Danielson component
2. The system uses the current prompts to generate a component summary
3. The user evaluates the quality of the summary and provides feedback
4. The data is saved to the Argilla database
5. The optimization process analyzes the evaluations and generates improved prompts
6. The cycle repeats with the optimized prompts, leading to better summaries over time

### Data Storage

All generated summaries, evaluations, and prompts are stored in Argilla datasets:
- `archer_outputs`: Contains the generated summaries
- `archer_evaluations`: Contains the evaluations of the summaries
- `archer_prompts`: Contains the prompts used to generate summaries

## Customization

You can customize the implementation by:

1. Adding more sample low-inference notes in `app.py`
2. Modifying the evaluation rubric in `app.py`
3. Adjusting the initial prompts in `app.py`
4. Changing the LLM models used for generation, evaluation, and optimization

## Troubleshooting

If you encounter issues:

1. Check that all environment variables are set correctly
2. Ensure the Argilla server is running if using database features
3. Check the logs for error messages
4. Try running with the `--no-archer` flag to test the basic functionality without optimization

## Contributing

Feel free to contribute to this implementation by:
1. Adding more comprehensive sample data
2. Improving the Gradio interface
3. Enhancing the evaluation rubric
4. Adding visualization for optimization progress 