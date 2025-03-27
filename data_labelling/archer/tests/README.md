# Archer System End-to-End Tests

This directory contains comprehensive end-to-end tests for the Archer system, which test the complete workflow from initialization through content generation, evaluation, and prompt optimization.

## Overview

The end-to-end tests are designed to verify that the entire system works correctly as an integrated whole. They simulate real user interactions and workflows, testing both through the Gradio interface and by directly using the Archer system components.

### Tests Included

1. **Gradio Interface Tests** (`test_end_to_end.py`)
   - Tests the system through the Gradio interface, following the complete user flow
   - Initializes the app, generates sample inputs, evaluates content, and optimizes prompts
   - Verifies that all steps in the workflow function correctly

2. **Direct Archer System Tests** (`test_archer_direct_flow.py`)
   - Tests the Archer system components directly without using the Gradio interface
   - Focuses on the core functionality: forward pass, backward pass, and database operations
   - Includes specific tests for database performance metrics and prompt retrieval

## Prerequisites

Before running the tests, ensure you have:

1. Set up your environment variables in a `.env` file in the project root with:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   ARGILLA_API_URL=your_argilla_url (e.g., http://localhost:6900)
   ARGILLA_API_KEY=your_argilla_key (e.g., admin.apikey)
   ```

2. Installed all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensured that the Argilla server is running and accessible at the URL specified in your `.env` file

## Running the Tests

You can run the tests using the provided runner script:

```bash
# Navigate to the tests directory
cd data_labelling/archer/tests

# Run all tests
python run_e2e_tests.py

# Run only the Gradio interface tests
python run_e2e_tests.py --gradio

# Run only the direct Archer system tests
python run_e2e_tests.py --direct
```

Alternatively, you can run individual test files directly:

```bash
# Run the Gradio interface test
python test_end_to_end.py

# Run the direct Archer system test
python test_archer_direct_flow.py
```

## Test Output

The tests produce detailed logging output that shows:

1. The initialization of system components
2. Each step in the testing process
3. Generated content and evaluation scores
4. Success or failure of prompt optimization
5. Test results and any failures

## Troubleshooting

If you encounter issues:

1. **Connection Errors**: Verify that your Argilla server is running and your API credentials are correct
2. **Missing Environment Variables**: Check that all required environment variables are set in your `.env` file
3. **Import Errors**: Ensure that your Python path includes all necessary directories
4. **API Rate Limits**: If you encounter rate limits with the LLM API, consider reducing the number of test runs

## Adding New Tests

When adding new end-to-end tests:

1. Follow the pattern of existing tests, with clear setup/teardown and step-by-step testing
2. Add appropriate assertions to verify each step's success
3. Use descriptive logging to make it clear what's being tested
4. Include your new test in the test runner script

## Test Coverage

These end-to-end tests focus on integration testing rather than unit testing. They verify that the system components work together correctly but do not test every possible edge case. For more comprehensive testing, consider adding:

1. Unit tests for individual components
2. Tests with various types of input data
3. Error handling and recovery tests
4. Performance and stress tests

## Note on Test Duration

Due to the nature of end-to-end testing with real API calls, these tests may take several minutes to complete. This is normal and expected behavior. 