"""
Tests for the refactored database schema with integrated prompts.
"""
import unittest
from unittest.mock import MagicMock, patch
import json
from archer.database.argilla import ArgillaDatabase

class TestDatabaseRefactoring(unittest.TestCase):
    """Test the refactored database schema with integrated prompts."""
    
    def setUp(self):
        """Set up test environment with mock Argilla client."""
        self.mock_argilla = MagicMock()
        self.patcher = patch('archer.database.argilla.rg.Argilla', return_value=self.mock_argilla)
        self.mock_rg = self.patcher.start()
        self.db = ArgillaDatabase(api_url="mock://test", api_key="mock_key")
        self.db.connect()
    
    def tearDown(self):
        """Clean up patches."""
        self.patcher.stop()
    
    def test_records_schema_has_prompt_fields(self):
        """Test that the records schema includes the prompt fields."""
        # For this test, we'll patch the Record creation and assert that
        # it has the correct fields when creating a record
        
        # Create a Record class mock
        mock_record = MagicMock()
        patcher = patch('archer.database.argilla.rg.Record', return_value=mock_record)
        mock_record_class = patcher.start()
        
        try:
            # Call store_record which will use our fields
            self.db.datasets = {'records': MagicMock()}
            self.db.store_record(
                input_data="test input",
                content="test content",
                generator_prompt="test generator prompt",
                evaluator_prompt="test evaluator prompt",
                prompt_generation=1,
                round_id="test_round"
            )
            
            # Verify Record constructor was called with the correct arguments
            mock_record_class.assert_called_once()
            call_args = mock_record_class.call_args
            args, kwargs = call_args
            
            # Check that fields include generator_prompt and evaluator_prompt
            fields = kwargs.get('fields', {})
            self.assertIn('generator_prompt', fields)
            self.assertIn('evaluator_prompt', fields)
            
            # Check that metadata includes prompt_generation
            metadata = kwargs.get('metadata', {})
            self.assertIn('prompt_generation', metadata)
        finally:
            patcher.stop()
    
    def test_store_record_includes_prompts(self):
        """Test that storing a record includes the prompt fields."""
        # Mock the dataset reference
        self.db.datasets = {'records': MagicMock()}
        
        # Call store_record with prompt information
        self.db.store_record(
            input_data="test input",
            content="test content",
            generator_prompt="test generator prompt",
            evaluator_prompt="test evaluator prompt",
            prompt_generation=1,
            round_id="test_round"
        )
        
        # Verify the record was created with prompt fields
        record_log_calls = self.db.datasets['records'].records.log.call_args_list
        self.assertEqual(len(record_log_calls), 1)
        records = record_log_calls[0][0][0]  # First call, first positional arg
        self.assertEqual(len(records), 1)
        record = records[0]
        
        # Check that the fields contain the prompts
        self.assertEqual(record.fields['generator_prompt'], "test generator prompt")
        self.assertEqual(record.fields['evaluator_prompt'], "test evaluator prompt")
        self.assertEqual(record.metadata['prompt_generation'], 1)
    
    def test_initialize_datasets_removes_prompt_datasets(self):
        """Test that initialize_datasets no longer creates prompt datasets."""
        # Create a partial mock that only mocks certain methods
        with patch.object(ArgillaDatabase, '_initialize_records_dataset', return_value=True):
            with patch.object(ArgillaDatabase, '_initialize_rounds_dataset', return_value=True):
                with patch.object(ArgillaDatabase, '_initialize_prompt_lineage_dataset', return_value=True):
                    with patch.object(ArgillaDatabase, '_initialize_outputs_dataset', return_value=True):
                        with patch.object(ArgillaDatabase, '_initialize_evaluations_dataset', return_value=True):
                            # Call initialize_datasets
                            result = self.db.initialize_datasets()
                            
                            # Verify it returns True (successful)
                            self.assertTrue(result)
                            
                            # Verify it didn't try to create the prompt datasets
                            self.assertNotIn('generator_prompts', self.db.datasets)
                            self.assertNotIn('evaluator_prompts', self.db.datasets)
    
    def test_get_prompts_from_records(self):
        """Test retrieving prompts from records."""
        # Mock dataset with sample records
        mock_dataset = MagicMock()
        mock_dataset.records.search.return_value = [
            MagicMock(fields={
                'generator_prompt': 'prompt1',
                'evaluator_prompt': 'eval_prompt1'
            }, metadata={'prompt_generation': 1}),
            MagicMock(fields={
                'generator_prompt': 'prompt2',
                'evaluator_prompt': 'eval_prompt2'
            }, metadata={'prompt_generation': 2})
        ]
        self.db.datasets = {'records': mock_dataset}
        
        # Test getting generator prompts
        generator_prompts = self.db.get_prompts_from_records(prompt_type="generator")
        self.assertEqual(len(generator_prompts), 2)
        self.assertEqual(generator_prompts[0]['content'], 'prompt1')
        self.assertEqual(generator_prompts[1]['content'], 'prompt2')
        
        # Test getting evaluator prompts
        evaluator_prompts = self.db.get_prompts_from_records(prompt_type="evaluator")
        self.assertEqual(len(evaluator_prompts), 2)
        self.assertEqual(evaluator_prompts[0]['content'], 'eval_prompt1')
        self.assertEqual(evaluator_prompts[1]['content'], 'eval_prompt2') 