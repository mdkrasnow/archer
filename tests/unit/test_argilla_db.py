import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
from datetime import datetime
import uuid

# Adjust path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_labelling.archer.database.supabase import SupabaseDatabase

class TestSupabaseDatabase(unittest.TestCase):
    def setUp(self):
        # Setup mocks and test database
        self.db = SupabaseDatabase(api_url="mock_url", api_key="mock_key")
        # Mock the Argilla client
        self.db.client = MagicMock()
        self.db.user_id = "test_user"
        # Setup test datasets
        self.db.datasets = {
            "records": MagicMock(),
            "generator_prompts": MagicMock(),
            "evaluator_prompts": MagicMock(),
            "rounds": MagicMock(),
            "prompt_lineage": MagicMock()
        }

    def test_connect(self):
        # Create a fresh database instance
        db = SupabaseDatabase(api_url="mock_url", api_key="mock_key")
        
        # Mock the Argilla client instantiation
        with patch('supabase.Argilla') as mock_supabase:
            # Setup mock user info
            mock_client = MagicMock()
            mock_user = MagicMock()
            mock_user.id = "test_user_id"
            mock_client.me = mock_user
            mock_supabase.return_value = mock_client
            
            # Test the connect method
            result = db.connect()
            
            # Assert the result and that the client was instantiated
            self.assertTrue(result)
            mock_supabase.assert_called_once_with(api_url="mock_url", api_key="mock_key")
            self.assertEqual(db.user_id, "test_user_id")
    
    def test_connect_error(self):
        # Create a fresh database instance
        db = SupabaseDatabase(api_url="mock_url", api_key="mock_key")
        
        # Mock the Argilla client to raise an exception
        with patch('supabase.Argilla', side_effect=Exception("Connection error")):
            # Test the connect method
            result = db.connect()
            
            # Assert the result
            self.assertFalse(result)

    def test_initialize_datasets(self):
        # Mock the helper methods to return True
        with patch.object(self.db, '_initialize_records_dataset', return_value=True) as mock_init_records, \
             patch.object(self.db, '_initialize_generator_prompts_dataset', return_value=True) as mock_init_gen, \
             patch.object(self.db, '_initialize_evaluator_prompts_dataset', return_value=True) as mock_init_eval, \
             patch.object(self.db, '_initialize_rounds_dataset', return_value=True) as mock_init_rounds, \
             patch.object(self.db, '_initialize_prompt_lineage_dataset', return_value=True) as mock_init_lineage:
            
            # Test the initialize_datasets method
            result = self.db.initialize_datasets()
            
            # Assert the result and that all helper methods were called
            self.assertTrue(result)
            mock_init_records.assert_called_once()
            mock_init_gen.assert_called_once()
            mock_init_eval.assert_called_once()
            mock_init_rounds.assert_called_once()
            mock_init_lineage.assert_called_once()
    
    def test_initialize_datasets_failure(self):
        # Mock one of the helper methods to return False
        with patch.object(self.db, '_initialize_records_dataset', return_value=True), \
             patch.object(self.db, '_initialize_generator_prompts_dataset', return_value=False), \
             patch.object(self.db, '_initialize_evaluator_prompts_dataset', return_value=True), \
             patch.object(self.db, '_initialize_rounds_dataset', return_value=True), \
             patch.object(self.db, '_initialize_prompt_lineage_dataset', return_value=True):
            
            # Test the initialize_datasets method
            result = self.db.initialize_datasets()
            
            # Assert the result
            self.assertFalse(result)
    
    def test_store_record(self):
        # Create test data
        input_data = "Test input"
        content = "Test content"
        generator_prompt_id = "gen-prompt-123"
        evaluator_prompt_id = "eval-prompt-456"
        round_id = "round-789"
        
        # Mock UUID generation
        test_uuid = "test-uuid-123"
        with patch('uuid.uuid4', return_value=MagicMock(return_value=test_uuid, __str__=lambda _: test_uuid)):
            # Mock the dataset's records.log method
            self.db.datasets["records"].records.log = MagicMock()
            
            # Test the store_record method
            result = self.db.store_record(input_data, content, generator_prompt_id, evaluator_prompt_id, round_id)
            
            # Assert the result and that records.log was called
            self.assertEqual(result, test_uuid)
            self.db.datasets["records"].records.log.assert_called_once()
    
    def test_store_generator_prompt(self):
        # Create test data
        content = "Test generator prompt"
        
        # Mock UUID generation
        test_uuid = "test-uuid-123"
        with patch('uuid.uuid4', return_value=MagicMock(return_value=test_uuid, __str__=lambda _: test_uuid)):
            # Mock the dataset's records.log method
            self.db.datasets["generator_prompts"].records.log = MagicMock()
            
            # Test the store_generator_prompt method
            result = self.db.store_generator_prompt(content)
            
            # Assert the result and that records.log was called
            self.assertEqual(result, test_uuid)
            self.db.datasets["generator_prompts"].records.log.assert_called_once()
    
    def test_store_evaluator_prompt(self):
        # Create test data
        content = "Test evaluator prompt"
        
        # Mock UUID generation
        test_uuid = "test-uuid-123"
        with patch('uuid.uuid4', return_value=MagicMock(return_value=test_uuid, __str__=lambda _: test_uuid)):
            # Mock the dataset's records.log method
            self.db.datasets["evaluator_prompts"].records.log = MagicMock()
            
            # Test the store_evaluator_prompt method
            result = self.db.store_evaluator_prompt(content)
            
            # Assert the result and that records.log was called
            self.assertEqual(result, test_uuid)
            self.db.datasets["evaluator_prompts"].records.log.assert_called_once()
    
    def test_create_round(self):
        # Create test data
        round_number = 1
        
        # Mock UUID generation
        test_uuid = "test-uuid-123"
        with patch('uuid.uuid4', return_value=MagicMock(return_value=test_uuid, __str__=lambda _: test_uuid)):
            # Mock the dataset's records.log method
            self.db.datasets["rounds"].records.log = MagicMock()
            
            # Test the create_round method
            result = self.db.create_round(round_number)
            
            # Assert the result and that records.log was called
            self.assertEqual(result, test_uuid)
            self.db.datasets["rounds"].records.log.assert_called_once()
    
    def test_update_record_evaluation(self):
        # Create test data
        record_id = "record-123"
        ai_score = 4
        ai_feedback = "Good job"
        ai_improved_output = "Better output"
        
        # Mock getting the record
        mock_record = MagicMock()
        mock_record.id = record_id
        self.db._get_record = MagicMock(return_value=mock_record)
        
        # Mock the dataset's records.log method
        self.db.datasets["records"].records.log = MagicMock()
        
        # Test the update_record_evaluation method
        result = self.db.update_record_evaluation(record_id, ai_score, ai_feedback, ai_improved_output)
        
        # Assert the result and that records.log was called
        self.assertTrue(result)
        self.db.datasets["records"].records.log.assert_called_once()
    
    def test_update_record_human_feedback(self):
        # Create test data
        record_id = "record-123"
        human_score = 3
        human_feedback = "Needs improvement"
        human_improved_output = "Much better output"
        
        # Mock getting the record
        mock_record = MagicMock()
        mock_record.id = record_id
        self.db._get_record = MagicMock(return_value=mock_record)
        
        # Mock the dataset's records.log method
        self.db.datasets["records"].records.log = MagicMock()
        
        # Test the update_record_human_feedback method
        result = self.db.update_record_human_feedback(record_id, human_score, human_feedback, human_improved_output)
        
        # Assert the result and that records.log was called
        self.assertTrue(result)
        self.db.datasets["records"].records.log.assert_called_once()
    
    def test_update_generator_prompt_performance(self):
        # Create test data
        prompt_id = "prompt-123"
        avg_score = 4.2
        rounds_survived = 2
        is_active = True
        
        # Mock getting the prompt record
        mock_filter = MagicMock()
        mock_query = MagicMock()
        mock_records = MagicMock()
        mock_records.to_list.return_value = [{"id": prompt_id}]
        
        with patch('supabase.Filter', return_value=mock_filter) as mock_filter_class, \
             patch('supabase.Query', return_value=mock_query) as mock_query_class:
            self.db.datasets["generator_prompts"].records.return_value = mock_records
            
            # Mock the dataset's records.log method
            self.db.datasets["generator_prompts"].records.log = MagicMock()
            
            # Test the update_generator_prompt_performance method
            result = self.db.update_generator_prompt_performance(prompt_id, avg_score, rounds_survived, is_active)
            
            # Assert the result and that records.log was called
            self.assertTrue(result)
            self.db.datasets["generator_prompts"].records.log.assert_called_once()
    
    def test_store_prompt_lineage(self):
        # Create test data
        parent_prompt_id = "parent-123"
        child_prompt_id = "child-456"
        round_id = "round-789"
        change_reason = "Performance improvement"
        
        # Mock UUID generation
        test_uuid = "test-uuid-123"
        with patch('uuid.uuid4', return_value=MagicMock(return_value=test_uuid, __str__=lambda _: test_uuid)):
            # Mock the dataset's records.log method
            self.db.datasets["prompt_lineage"].records.log = MagicMock()
            
            # Test the store_prompt_lineage method
            result = self.db.store_prompt_lineage(parent_prompt_id, child_prompt_id, round_id, change_reason)
            
            # Assert the result and that records.log was called
            self.assertEqual(result, test_uuid)
            self.db.datasets["prompt_lineage"].records.log.assert_called_once()
    
    def test_get_current_data_for_annotation(self):
        # Create test data
        round_id = "round-123"
        limit = 10
        
        # Mock the records
        mock_filter = MagicMock()
        mock_query = MagicMock()
        mock_records = MagicMock()
        
        # Sample record data
        test_records = [
            {
                "id": "record-1",
                "fields": {
                    "input": "Test input 1",
                    "content": "Test content 1"
                },
                "responses": [
                    {"question": {"name": "ai_score"}, "value": 4},
                    {"question": {"name": "ai_feedback"}, "value": "Good job"},
                    {"question": {"name": "ai_improved_output"}, "value": "Better content"}
                ],
                "metadata": {
                    "generator_prompt_id": "gen-prompt-1",
                    "evaluator_prompt_id": "eval-prompt-1",
                    "is_validated": "False"
                }
            },
            {
                "id": "record-2",
                "fields": {
                    "input": "Test input 2",
                    "content": "Test content 2"
                },
                "responses": [
                    {"question": {"name": "ai_score"}, "value": 3},
                    {"question": {"name": "ai_feedback"}, "value": "Needs work"},
                    {"question": {"name": "ai_improved_output"}, "value": "Improved content"},
                    {"question": {"name": "human_score"}, "value": 4},
                    {"question": {"name": "human_feedback"}, "value": "Human feedback"},
                    {"question": {"name": "human_improved_output"}, "value": "Human improved content"}
                ],
                "metadata": {
                    "generator_prompt_id": "gen-prompt-2",
                    "evaluator_prompt_id": "eval-prompt-2",
                    "is_validated": "True"
                }
            }
        ]
        
        mock_records.to_list.return_value = test_records
        
        with patch('supabase.Filter', return_value=mock_filter) as mock_filter_class, \
             patch('supabase.Query', return_value=mock_query) as mock_query_class:
            self.db.datasets["records"].records.return_value = mock_records
            
            # Test the get_current_data_for_annotation method
            result = self.db.get_current_data_for_annotation(round_id, limit)
            
            # Assert the result has the expected structure
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertIn("record_id", result.columns)
            self.assertIn("input", result.columns)
            self.assertIn("content", result.columns)
            self.assertIn("ai_score", result.columns)
    
    def test_get_performance_metrics(self):
        # Mock the datasets responses
        self.db.datasets["records"].records = MagicMock()
        self.db.datasets["generator_prompts"].records = MagicMock()
        self.db.datasets["rounds"].records = MagicMock()
        
        # Sample data
        test_records = [
            {
                "id": "record-1",
                "responses": [
                    {"question": {"name": "ai_score"}, "value": 4}
                ],
                "metadata": {
                    "generator_prompt_id": "gen-prompt-1",
                    "round_id": "round-1"
                }
            },
            {
                "id": "record-2",
                "responses": [
                    {"question": {"name": "ai_score"}, "value": 3}
                ],
                "metadata": {
                    "generator_prompt_id": "gen-prompt-2",
                    "round_id": "round-1"
                }
            }
        ]
        
        test_gen_prompts = [
            {
                "id": "gen-prompt-1",
                "fields": {"content": "Prompt 1"},
                "metadata": {
                    "average_score": "4.0",
                    "rounds_survived": "2",
                    "parent_prompt_id": "root"
                }
            },
            {
                "id": "gen-prompt-2",
                "fields": {"content": "Prompt 2"},
                "metadata": {
                    "average_score": "3.0",
                    "rounds_survived": "1",
                    "parent_prompt_id": "gen-prompt-1"
                }
            }
        ]
        
        test_rounds = [
            {
                "id": "round-1",
                "fields": {"number": "1", "status": "completed"},
                "metadata": {
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat()
                }
            }
        ]
        
        self.db.datasets["records"].records().to_list.return_value = test_records
        self.db.datasets["generator_prompts"].records().to_list.return_value = test_gen_prompts
        self.db.datasets["rounds"].records().to_list.return_value = test_rounds
        
        # Test the get_performance_metrics method
        result = self.db.get_performance_metrics()
        
        # Assert the result has the expected structure
        self.assertIsInstance(result, dict)
        self.assertIn("rounds", result)
        self.assertIn("prompts", result)
        self.assertIn("scores", result)
        
    def test_get_prompt_history(self):
        # Mock the generator_prompts dataset response
        self.db.datasets["generator_prompts"].records = MagicMock()
        
        # Sample prompt data
        test_prompts = [
            {
                "id": "prompt-1",
                "fields": {"content": "Test prompt 1"},
                "metadata": {
                    "prompt_id": "prompt-1",
                    "parent_prompt_id": "root",
                    "average_score": "4.2",
                    "rounds_survived": "2",
                    "version": "1",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "id": "prompt-2",
                "fields": {"content": "Test prompt 2"},
                "metadata": {
                    "prompt_id": "prompt-2",
                    "parent_prompt_id": "prompt-1",
                    "average_score": "3.8",
                    "rounds_survived": "1",
                    "version": "2",
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
        
        self.db.datasets["generator_prompts"].records().to_list.return_value = test_prompts
        
        # Test the get_prompt_history method
        result = self.db.get_prompt_history()
        
        # Assert the result is a DataFrame with the expected structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("prompt_id", result.columns)
        self.assertIn("parent_prompt_id", result.columns)
        self.assertIn("content", result.columns)
        
    def test_get_active_evaluator_prompts(self):
        # Mock the evaluator_prompts dataset response
        self.db.datasets["evaluator_prompts"].records = MagicMock()
        
        # Sample evaluator prompt data
        test_evaluator_prompts = [
            {
                "id": "eval-prompt-1",
                "fields": {"content": "Test evaluator prompt 1"},
                "metadata": {
                    "prompt_id": "eval-prompt-1",
                    "is_active": "True",
                    "version": "1",
                    "created_at": datetime.now().isoformat()
                }
            },
            {
                "id": "eval-prompt-2",
                "fields": {"content": "Test evaluator prompt 2"},
                "metadata": {
                    "prompt_id": "eval-prompt-2",
                    "is_active": "False",
                    "version": "2",
                    "created_at": datetime.now().isoformat()
                }
            }
        ]
        
        # Create mock Filter and Query
        mock_filter = MagicMock()
        mock_query = MagicMock()
        mock_records = MagicMock()
        mock_records.to_list.return_value = [test_evaluator_prompts[0]]  # Only return active prompts
        
        with patch('supabase.Filter', return_value=mock_filter) as mock_filter_class, \
             patch('supabase.Query', return_value=mock_query) as mock_query_class:
            self.db.datasets["evaluator_prompts"].records.return_value = mock_records
            
            # Test the get_active_evaluator_prompts method
            result = self.db.get_active_evaluator_prompts()
            
            # Assert the result has the expected structure
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], "eval-prompt-1")
    
    def test_get_round_metrics(self):
        # Create test data
        round_id = "round-123"
        
        # Mock getting the round
        mock_round = {
            "id": round_id,
            "fields": {"number": "1", "status": "completed"},
            "metadata": {
                "metrics": '{"average_score": 4.2, "total_records": 10}'
            }
        }
        
        # Mock Filter and Query
        mock_filter = MagicMock()
        mock_query = MagicMock()
        mock_records = MagicMock()
        mock_records.to_list.return_value = [mock_round]
        
        with patch('supabase.Filter', return_value=mock_filter) as mock_filter_class, \
             patch('supabase.Query', return_value=mock_query) as mock_query_class:
            self.db.datasets["rounds"].records.return_value = mock_records
            
            # Test the get_round_metrics method
            result = self.db.get_round_metrics(round_id)
            
            # Assert the result has the expected structure
            self.assertIsInstance(result, dict)
            self.assertIn("average_score", result)
            self.assertIn("total_records", result)
            
    def test_get_validated_evaluations(self):
        """Test the get_validated_evaluations method with human-validated evaluations."""
        # Add "evaluations" to the datasets dictionary
        self.db.datasets["evaluations"] = MagicMock()
        
        # Create mock records for the evaluations
        mock_records = [
            {
                "metadata": {
                    "output_id": "output-123",
                    "prompt_id": "prompt-456",
                    "is_human": "1",
                    "timestamp": "2025-03-27T01:00:00Z"
                },
                "fields": {
                    "input": "Test input 1",
                    "generated_content": "Test content 1"
                },
                "responses": [
                    {
                        "question": {"name": "score"},
                        "value": "4"
                    },
                    {
                        "question": {"name": "feedback"},
                        "value": "Good output"
                    },
                    {
                        "question": {"name": "improved_output"},
                        "value": "Better output 1"
                    }
                ]
            },
            {
                "metadata": {
                    "output_id": "output-789",
                    "prompt_id": "prompt-456",
                    "is_human": "1",
                    "timestamp": "2025-03-27T02:00:00Z" 
                },
                "fields": {
                    "input": "Test input 2",
                    "generated_content": "Test content 2"
                },
                "responses": [
                    {
                        "question": {"name": "score"},
                        "value": "3"
                    },
                    {
                        "question": {"name": "feedback"},
                        "value": "Average output"
                    },
                    {
                        "question": {"name": "improved_output"},
                        "value": "Better output 2"
                    }
                ]
            }
        ]
        
        # Mock the Filter, Query and records methods
        mock_filter = MagicMock()
        mock_query = MagicMock()
        mock_records_result = MagicMock()
        mock_records_result.to_list.return_value = mock_records
        
        with patch('supabase.Filter', return_value=mock_filter) as mock_filter_class, \
             patch('supabase.Query', return_value=mock_query) as mock_query_class:
            # Set up the mock records result
            self.db.datasets["evaluations"].records.return_value = mock_records_result
            
            # Test with default limit
            result = self.db.get_validated_evaluations()
            
            # Verify the filter was created correctly
            mock_filter_class.assert_called_once_with(("metadata.is_human", "==", "1"))
            mock_query_class.assert_called_once_with(filter=mock_filter)
            
            # Check the result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            
            # Check the values in the DataFrame
            self.assertEqual(result.iloc[0]["output_id"], "output-789")  # Should be first because of timestamp sort
            self.assertEqual(result.iloc[1]["output_id"], "output-123")
            self.assertEqual(result.iloc[0]["score"], 3.0)
            self.assertEqual(result.iloc[1]["score"], 4.0)
            
            # Test with custom limit
            result = self.db.get_validated_evaluations(limit=1)
            
            # Check that limit is respected
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["output_id"], "output-789")  # Only get the newest one
            
    def test_get_validated_evaluations_empty(self):
        """Test the get_validated_evaluations method when no validated evaluations exist."""
        # Add "evaluations" to the datasets dictionary
        self.db.datasets["evaluations"] = MagicMock()
        
        # Mock empty records result
        mock_filter = MagicMock()
        mock_query = MagicMock()
        mock_records_result = MagicMock()
        mock_records_result.to_list.return_value = []
        
        with patch('supabase.Filter', return_value=mock_filter), \
             patch('supabase.Query', return_value=mock_query):
            # Set up the mock records result
            self.db.datasets["evaluations"].records.return_value = mock_records_result
            
            # Test the method
            result = self.db.get_validated_evaluations()
            
            # Check the result
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)
            
    def test_get_validated_evaluations_error(self):
        """Test the get_validated_evaluations method when an error occurs."""
        # Add "evaluations" to the datasets dictionary
        self.db.datasets["evaluations"] = MagicMock()
        
        # Mock an error in the query
        with patch('supabase.Filter', side_effect=Exception("Test error")):
            # Test the method
            result = self.db.get_validated_evaluations()
            
            # Check the result
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main() 