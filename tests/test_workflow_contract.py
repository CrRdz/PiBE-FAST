"""Bounded workflow regression matrix; expected outputs do not call the reducer."""
import unittest
from scripts.verify_workflow_contracts import run_cases

class WorkflowContractTest(unittest.TestCase):
    def test_state_transitions_and_representation_contract(self):
        rows = run_cases()
        self.assertTrue(rows)
        self.assertTrue(all(row['passed'] for row in rows))
