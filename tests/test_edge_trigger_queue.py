import tempfile
import unittest
from pathlib import Path

from app.edge_trigger_queue import EdgeTriggerQueue


class EdgeTriggerQueueTest(unittest.TestCase):
    def test_offline_trigger_survives_restart_and_reconnect_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edge-triggers.json"
            queue = EdgeTriggerQueue(path)
            trigger = queue.enqueue(
                "passive_balance", "sustained_personal_balance_change", created_at=10.0
            )
            queue.mark_delivery_attempt(trigger["id"])

            restarted = EdgeTriggerQueue(path)
            pending = restarted.unresolved()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["attempts"], 1)
            self.assertEqual(pending[0]["state"], "unresolved")

            acknowledged = restarted.acknowledge(
                trigger["id"], acknowledged_at=15.0
            )
            self.assertEqual(acknowledged["state"], "acknowledged")
            self.assertEqual(restarted.unresolved(), [])

            reloaded = EdgeTriggerQueue(path)
            self.assertEqual(reloaded.unresolved(), [])

    def test_no_response_escalates_operational_state_without_medical_urgency(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = EdgeTriggerQueue(
                Path(directory) / "edge-triggers.json",
                repeat_prompt_seconds=30.0,
                attention_required_seconds=120.0,
            )
            trigger = queue.enqueue("passive", "change", created_at=10.0)

            pending = queue.evaluate_timeouts(now=39.9)[0]
            repeated = queue.evaluate_timeouts(now=40.0)[0]
            attention = queue.evaluate_timeouts(now=130.0)[0]

            self.assertEqual(pending["operational_level"], "pending_response")
            self.assertEqual(repeated["operational_level"], "repeat_prompt_due")
            self.assertEqual(
                attention["operational_level"], "caregiver_attention_required"
            )
            self.assertNotIn("urgency", attention)
            self.assertEqual(attention["attempts"], 0)
            self.assertIsNotNone(attention["prompt_due_recorded_at"])
            self.assertIsNotNone(attention["attention_required_recorded_at"])

            queue.acknowledge(trigger["id"], acknowledged_at=140.0)
            self.assertEqual(queue.evaluate_timeouts(now=200.0), [])

    def test_timeout_configuration_rejects_inverted_deadlines(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                EdgeTriggerQueue(
                    Path(directory) / "edge-triggers.json",
                    repeat_prompt_seconds=30.0,
                    attention_required_seconds=20.0,
                )

    def test_first_late_evaluation_records_both_elapsed_milestones(self):
        with tempfile.TemporaryDirectory() as directory:
            queue = EdgeTriggerQueue(Path(directory) / "edge-triggers.json")
            queue.enqueue("scheduled", "due", created_at=0.0)

            record = queue.evaluate_timeouts(now=130.0)[0]

            self.assertEqual(
                record["operational_level"], "caregiver_attention_required"
            )
            self.assertIsNotNone(record["prompt_due_recorded_at"])
            self.assertIsNotNone(record["attention_required_recorded_at"])


if __name__ == "__main__":
    unittest.main()
