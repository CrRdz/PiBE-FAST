import unittest
import numpy as np
from training.multimodal.train_fusion import _standardize, _participant_predictions

class FusionTrainingTest(unittest.TestCase):
    def test_validation_outlier_cannot_change_training_scaler(self):
        train = np.array([[0.], [2.]])
        fitted, validation, means, scales = _standardize(train, np.array([[1000.]]))
        np.testing.assert_allclose(means, [1.])
        np.testing.assert_allclose(scales, [1.])
        np.testing.assert_allclose(fitted[:,0], [-1., 1.])
        np.testing.assert_allclose(validation, [[999.]])

    def test_participant_aggregation_rejects_incompatible_outcomes(self):
        y, p = _participant_predictions(np.array([0,0,1]), np.array([.1,.3,.8]), np.array(['a','a','b']))
        np.testing.assert_allclose(p, [.2,.8])
        with self.assertRaises(ValueError):
            _participant_predictions(np.array([0,1]), np.array([.1,.8]), np.array(['a','a']))
