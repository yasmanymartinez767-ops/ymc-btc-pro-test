import unittest
from ai_display import present_estimate

class TestDisplay(unittest.TestCase):
    def test_nan_not_displayed(self):
        r = present_estimate({'signal':'ALCISTA','validated':False,'validation_accuracy':float('nan'),'validation_samples':0})
        self.assertEqual(r['signal'],'ESPERAR')
        self.assertNotIn('nan',r['detail'].lower())
    def test_missing_result(self):
        self.assertEqual(present_estimate(None)['status'],'SIN DATOS')
    def test_false_validation(self):
        r=present_estimate({'signal':'BAJISTA','validated':True,'validation_accuracy':1.7,'validation_samples':300})
        self.assertEqual(r['signal'],'ESPERAR')
    def test_valid_contract(self):
        r=present_estimate({'signal':'BAJISTA','validated':True,'validation_accuracy':.54,'validation_samples':200})
        self.assertEqual(r['status'],'VALIDADO')
        self.assertIn('54.0%',r['detail'])
    def test_small_sample(self):
        r=present_estimate({'signal':'ALCISTA','validated':True,'validation_accuracy':.9,'validation_samples':5})
        self.assertEqual(r['signal'],'ESPERAR')

if __name__=='__main__': unittest.main()
