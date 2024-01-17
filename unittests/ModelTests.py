#!/usr/bin/env python
"""
model tests
"""


import unittest

## import model specific functions and variables
## model isnt in the same directory as this file, so we need to go up a level
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(data_dir="cs-train", test=True)
        self.assertTrue(os.path.exists("models/test-all-0_1.joblib"))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        all_data, all_models = model_load(data_dir="cs-train")
        model = all_models['netherlands']
        
        # ensure that the model object has the correct structure
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load()
    
        ## ensure that a list can be passed
        query = {'country': 'netherlands',
                 'year': '2018',
                 'month': '1',
                 'day': '5'
        }

        result = model_predict(country=query['country'],
                               year=query['year'],
                               month=query['month'],
                               day=query['day'],
                               test=True)
        y_pred = result['y_pred']
        self.assertTrue(y_pred[0] > 0.0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()