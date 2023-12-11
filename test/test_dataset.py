import unittest
from dataset import Dataset

class DatasetTestCase(unittest.TestCase):
    def setUp(self):
        # Set up the dataset for testing
        self.dataset = Dataset(train_file="../data/train_1K.sdf",
                               val_file="../data/D2_7jvr_dop_393b_2comp_final_10M_test_10K.sdf",
                               test_file="../data/test_10K.sdf",
                               shuffle=False,
                               split_ratio=0.8,
                               test_size="1M")

    def test_train_data(self):
        # Test if the train data is loaded correctly
        train_data = self.dataset.get_train_data()
        self.assertIsNotNone(train_data)
        self.assertEqual(len(train_data), 8000)  # Assuming 80% of 10K is 8000

    def test_val_data(self):
        # Test if the validation data is loaded correctly
        val_data = self.dataset.get_val_data()
        self.assertIsNotNone(val_data)
        self.assertEqual(len(val_data), 2000)  # Assuming 20% of 10K is 2000

    def test_test_data(self):
        # Test if the test data is loaded correctly
        test_data = self.dataset.get_test_data()
        self.assertIsNotNone(test_data)
        self.assertEqual(len(test_data), 1000000)  # Assuming test_size is "1M"

if __name__ == '__main__':
    unittest.main()