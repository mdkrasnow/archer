#!/usr/bin/env python
"""
run_e2e_tests.py

A script to run the end-to-end tests for the Archer system.
This allows running either all tests or specific test types.
"""

import os
import sys
import argparse
import unittest
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure the parent directories are in the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_environment():
    """
    Load environment variables from .env file.
    """
    # Try to load from .env file
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)

    # Check for required environment variables
    required_vars = ['OPENROUTER_API_KEY', 'ARGILLA_API_URL', 'ARGILLA_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Some tests may fail due to missing credentials.")
        return False
    return True

def run_gradio_interface_tests():
    """
    Run the tests that use the Gradio interface.
    """
    logger.info("Running Gradio interface end-to-end tests...")
    from test_end_to_end import TestEndToEnd
    
    # Create a test suite with just the Gradio tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEnd)
    
    # Run the tests and return the result
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

def run_direct_archer_tests():
    """
    Run the tests that directly use the Archer system without Gradio.
    """
    logger.info("Running direct Archer system end-to-end tests...")
    from test_archer_direct_flow import TestArcherDirectFlow
    
    # Create a test suite with just the direct Archer tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestArcherDirectFlow)
    
    # Run the tests and return the result
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

def run_all_tests():
    """
    Run all end-to-end tests.
    """
    logger.info("Running all end-to-end tests...")
    
    # Import the test cases
    from test_end_to_end import TestEndToEnd
    from test_archer_direct_flow import TestArcherDirectFlow
    
    # Create a test suite with all tests
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEndToEnd))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestArcherDirectFlow))
    
    # Run the tests and return the result
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

def main():
    """
    Main function to parse arguments and run tests.
    """
    parser = argparse.ArgumentParser(description="Run end-to-end tests for the Archer system")
    parser.add_argument("--gradio", action="store_true", help="Run only Gradio interface tests")
    parser.add_argument("--direct", action="store_true", help="Run only direct Archer system tests")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    args = parser.parse_args()
    
    # Load environment variables
    env_loaded = load_environment()
    if not env_loaded:
        logger.warning("Environment setup incomplete. Some tests may fail.")
    
    # Determine which tests to run
    if args.gradio:
        success = run_gradio_interface_tests()
    elif args.direct:
        success = run_direct_archer_tests()
    else:  # Default to all tests
        success = run_all_tests()
    
    # Return exit code based on test results
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 