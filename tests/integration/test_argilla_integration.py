import unittest
import os
import sys
import time
import pandas as pd
from datetime import datetime
import uuid
import json

# Adjust path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_labelling.archer.database.argilla import ArgillaDatabase

@unittest.skipIf(not os.getenv("TEST_ARGILLA_API_URL"), "Integration tests require TEST_ARGILLA_API_URL env var")
class TestArgillaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup real database connection
        cls.db = ArgillaDatabase(
            api_url=os.getenv("TEST_ARGILLA_API_URL"),
            api_key=os.getenv("TEST_ARGILLA_API_KEY", "admin.apikey")
        )
        # Ensure connection
        connected = cls.db.connect()
        if not connected:
            raise unittest.SkipTest("Could not connect to Argilla server")
        
        # Initialize datasets
        initialized = cls.db.initialize_datasets()
        if not initialized:
            raise unittest.SkipTest("Could not initialize datasets")
            
        # Store test IDs for cleanup
        cls.test_ids = {
            "records": [],
            "generator_prompts": [],
            "evaluator_prompts": [],
            "rounds": [],
            "prompt_lineage": []
        }
        
    def setUp(self):
        # Add unique prefix for this test run to prevent collisions
        self.test_prefix = f"test_{int(time.time())}_{uuid.uuid4().hex[:8]}_"
        
    def tearDown(self):
        # Clean up could be implemented if Argilla provides a way to delete records
        # For now, we'll just log which records were created
        pass
        
    @classmethod
    def tearDownClass(cls):
        # Log which records would need to be cleaned up in a real environment
        print(f"Test created the following records that should be cleaned up:")
        for dataset, ids in cls.test_ids.items():
            if ids:
                print(f"  {dataset}: {', '.join(ids)}")
    
    def test_end_to_end_workflow(self):
        """Test the full workflow from storing prompts to retrieving metrics."""
        # 1. Create generator prompt
        generator_prompt = f"{self.test_prefix}Test generator prompt"
        generator_prompt_id = self.db.store_generator_prompt(generator_prompt)
        self.assertIsNotNone(generator_prompt_id)
        self.test_ids["generator_prompts"].append(generator_prompt_id)
        
        # 2. Create evaluator prompt
        evaluator_prompt = f"{self.test_prefix}Test evaluator prompt"
        evaluator_prompt_id = self.db.store_evaluator_prompt(evaluator_prompt)
        self.assertIsNotNone(evaluator_prompt_id)
        self.test_ids["evaluator_prompts"].append(evaluator_prompt_id)
        
        # 3. Create round
        round_number = 1
        round_id = self.db.create_round(round_number)
        self.assertIsNotNone(round_id)
        self.test_ids["rounds"].append(round_id)
        
        # 4. Store record (generated content)
        input_data = f"{self.test_prefix}Test input"
        content = f"{self.test_prefix}Test content"
        record_id = self.db.store_record(
            input_data, 
            content, 
            generator_prompt_id, 
            evaluator_prompt_id, 
            round_id
        )
        self.assertIsNotNone(record_id)
        self.test_ids["records"].append(record_id)
        
        # 5. Update record with AI evaluation
        ai_score = 4
        ai_feedback = f"{self.test_prefix}Good content, but could be improved"
        ai_improved_output = f"{self.test_prefix}Improved test content"
        ai_eval_success = self.db.update_record_evaluation(
            record_id,
            ai_score,
            ai_feedback,
            ai_improved_output
        )
        self.assertTrue(ai_eval_success)
        
        # 6. Update record with human feedback
        human_score = 3
        human_feedback = f"{self.test_prefix}Needs more specific details"
        human_improved_output = f"{self.test_prefix}Better improved test content"
        human_feedback_success = self.db.update_record_human_feedback(
            record_id,
            human_score,
            human_feedback,
            human_improved_output
        )
        self.assertTrue(human_feedback_success)
        
        # 7. Update generator prompt performance
        avg_score = 3.5
        rounds_survived = 1
        is_active = True
        update_success = self.db.update_generator_prompt_performance(
            generator_prompt_id,
            avg_score,
            rounds_survived,
            is_active
        )
        self.assertTrue(update_success)
        
        # 8. Create a child prompt and store lineage
        child_prompt = f"{self.test_prefix}Child generator prompt"
        child_prompt_id = self.db.store_generator_prompt(
            child_prompt,
            parent_prompt_id=generator_prompt_id,
            version=2
        )
        self.assertIsNotNone(child_prompt_id)
        self.test_ids["generator_prompts"].append(child_prompt_id)
        
        change_reason = f"{self.test_prefix}Improved based on feedback"
        lineage_id = self.db.store_prompt_lineage(
            generator_prompt_id,
            child_prompt_id,
            round_id,
            change_reason
        )
        self.assertIsNotNone(lineage_id)
        self.test_ids["prompt_lineage"].append(lineage_id)
        
        # 9. Get data for annotation
        df = self.db.get_current_data_for_annotation(round_id)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 1)
        
        # 10. Get performance metrics
        metrics = self.db.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
        
        # 11. Get prompt history
        history = self.db.get_prompt_history()
        self.assertIsInstance(history, pd.DataFrame)
        self.assertGreaterEqual(len(history), 2)  # Original + child prompt
        
        # 12. Get active evaluator prompts
        active_prompts = self.db.get_active_evaluator_prompts()
        self.assertIsInstance(active_prompts, list)
        
        # 13. Complete the round and update metrics
        round_metrics = {
            "average_score": 3.5,
            "total_records": 1,
            "completed_records": 1
        }
        update_round_success = self.db.update_round(
            round_id,
            status="completed",
            metrics=round_metrics
        )
        self.assertTrue(update_round_success)
        
        # 14. Get round metrics
        retrieved_metrics = self.db.get_round_metrics(round_id)
        self.assertIsInstance(retrieved_metrics, dict)
        self.assertEqual(retrieved_metrics.get("average_score"), 3.5)
    
    def test_store_and_retrieve_record(self):
        """Test storing and retrieving a single record."""
        # 1. Create generator prompt
        generator_prompt = f"{self.test_prefix}Test generator prompt for record"
        generator_prompt_id = self.db.store_generator_prompt(generator_prompt)
        self.test_ids["generator_prompts"].append(generator_prompt_id)
        
        # 2. Create evaluator prompt
        evaluator_prompt = f"{self.test_prefix}Test evaluator prompt for record"
        evaluator_prompt_id = self.db.store_evaluator_prompt(evaluator_prompt)
        self.test_ids["evaluator_prompts"].append(evaluator_prompt_id)
        
        # 3. Create round
        round_number = 1
        round_id = self.db.create_round(round_number)
        self.test_ids["rounds"].append(round_id)
        
        # 4. Store record
        input_data = f"{self.test_prefix}Test input for record retrieval"
        content = f"{self.test_prefix}Test content for record retrieval"
        record_id = self.db.store_record(
            input_data, 
            content, 
            generator_prompt_id, 
            evaluator_prompt_id, 
            round_id
        )
        self.test_ids["records"].append(record_id)
        
        # 5. Retrieve the record using _get_record
        record = self.db._get_record(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.get("metadata", {}).get("generator_prompt_id"), generator_prompt_id)
        
        # Check field content matches what we stored
        fields = record.get("fields", {})
        self.assertEqual(fields.get("input"), input_data)
        self.assertEqual(fields.get("content"), content)
    
    def test_prompt_lineage_tracking(self):
        """Test tracking prompt lineage across generations."""
        # 1. Create parent prompt
        parent_prompt = f"{self.test_prefix}Parent prompt"
        parent_id = self.db.store_generator_prompt(parent_prompt)
        self.test_ids["generator_prompts"].append(parent_id)
        
        # 2. Create round
        round_id = self.db.create_round(1)
        self.test_ids["rounds"].append(round_id)
        
        # 3. Create child prompts (generation 1)
        child_prompts = []
        for i in range(3):
            child_prompt = f"{self.test_prefix}Child prompt {i}"
            child_id = self.db.store_generator_prompt(
                child_prompt,
                parent_prompt_id=parent_id,
                version=i+1
            )
            self.test_ids["generator_prompts"].append(child_id)
            child_prompts.append(child_id)
            
            # Store lineage
            lineage_id = self.db.store_prompt_lineage(
                parent_id,
                child_id,
                round_id,
                f"{self.test_prefix}Variation {i+1}"
            )
            self.test_ids["prompt_lineage"].append(lineage_id)
        
        # 4. Create grandchild prompt (generation 2)
        grandchild_prompt = f"{self.test_prefix}Grandchild prompt"
        grandchild_id = self.db.store_generator_prompt(
            grandchild_prompt,
            parent_prompt_id=child_prompts[0],
            version=1
        )
        self.test_ids["generator_prompts"].append(grandchild_id)
        
        # Store lineage
        lineage_id = self.db.store_prompt_lineage(
            child_prompts[0],
            grandchild_id,
            round_id,
            f"{self.test_prefix}Further improvement"
        )
        self.test_ids["prompt_lineage"].append(lineage_id)
        
        # 5. Get prompt history and check lineage
        history = self.db.get_prompt_history()
        
        # Should have at least our 5 prompts (parent + 3 children + grandchild)
        self.assertGreaterEqual(len(history), 5)
        
        # Check if our parent-child relationships are captured
        parent_row = history[history["prompt_id"] == parent_id]
        self.assertEqual(len(parent_row), 1)
        
        # Check for grandchild
        grandchild_row = history[history["prompt_id"] == grandchild_id]
        self.assertEqual(len(grandchild_row), 1)
        self.assertEqual(grandchild_row.iloc[0]["parent_id"], child_prompts[0])
        
        # Get lineage data directly
        lineage_data = self.db.get_prompt_lineage()
        self.assertIsInstance(lineage_data, pd.DataFrame)
        self.assertGreaterEqual(len(lineage_data), 4)  # Our 4 lineage records
        
if __name__ == "__main__":
    unittest.main() 