"""Synthetic-only unit tests for reconstructed XGBoost pipeline."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from xgb_shap_ablation import features,score,evaluate

class TestPipeline(unittest.TestCase):
    def test_features(self):
        x=np.ones((4,60,6))
        x[0,:3,0]=np.nan
        values,names=features(x)
        self.assertEqual(values.shape,(4,24))
        self.assertEqual(names[0],"eda_mean")
        self.assertTrue(np.isfinite(values).all())

    def test_single_class_auc(self):
        self.assertIsNone(score(np.zeros(3,dtype=int),np.array([.1,.2,.3]))["roc_auc"])

    def test_loso_ablation(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(24,60,6))
        y=np.tile([0,1,0,1,0,1],4)
        groups=np.repeat(["s1","s2","s3","s4"],6)
        result=evaluate(x,y,groups)
        self.assertEqual(len(result["folds"]),4)
        self.assertEqual(len(result["folds"][0]["ablation"]),6)

if __name__=="__main__":
    unittest.main()
