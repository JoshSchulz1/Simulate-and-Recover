
import unittest
import pandas as pd 

class TestSimulation(unittest.TestCase):
    def test_simulation_output(self):
        df = pd.read_csv("results/output.csv")
        self.assertFalse(df.empty)
        self.assertTrue("N" in df.columns)

if __name__ == "__main__":
    unittest.main()
