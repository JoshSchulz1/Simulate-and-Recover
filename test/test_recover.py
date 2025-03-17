import unittest
import pandas as pd 

class TestRecovery(unittest.TestCase):
    def test_recovery_output(self):
        df = pd.read_csv("results/recovered.csv")
        self.assertFalse(df.empty)
        self.assertTrue("v_est" in df.columns)

if __name__ == "__main__":
    unittest.main()
