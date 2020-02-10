import unittest
import numpy as np

from eva_storage.temporal_filters.src.uniform_sampling import UniformSampling


class UniformSamplingTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_uniform_sampling_np(self):
        x = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
        y = UniformSampling.sample(x, 1)
        z = UniformSampling.sample(x, 2)
        self.assertTrue(np.array_equal(x,y))
        self.assertFalse(np.array_equal(x,z))

    def test_uniform_sampling_list(self):
        x = [1,2,3,4,5,6,7,8,9,0]
        y = UniformSampling.sample(x,1)
        z = UniformSampling.sample(x,2)
        self.assertEqual(x,y)
        self.assertNotEqual(x,z)






if __name__ == '__main__':

    unittest.main()
