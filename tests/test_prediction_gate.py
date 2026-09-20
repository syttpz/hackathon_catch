import unittest

from motion.prediction_gate import PredictionGate


TARGET = [100.0, 200.0, 300.0]


class PredictionGateTests(unittest.TestCase):
    def test_stable_prediction_and_reset(self):
        gate = PredictionGate()
        results = [gate.add(index * 0.02, TARGET, 10.0) for index in range(12)]
        self.assertFalse(results[0])
        self.assertTrue(results[-1])

        gate.reset()
        self.assertFalse(gate.add(0.25, TARGET, 10.0))

    def test_drifting_arrival_is_rejected(self):
        gate = PredictionGate()
        results = [
            gate.add(index * 0.02, TARGET, 10.0 + index * 0.1)
            for index in range(12)
        ]
        self.assertFalse(any(results))

    def test_large_gap_restarts_confirmation(self):
        gate = PredictionGate()
        for index in range(12):
            gate.add(index * 0.02, TARGET, 10.0)
        self.assertFalse(gate.add(1.0, TARGET, 10.0))


if __name__ == "__main__":
    unittest.main()
