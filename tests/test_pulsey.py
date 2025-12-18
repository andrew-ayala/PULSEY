import unittest
import numpy as np
import PULSEY as p
from astropy.table import Table

star = p.star(lmModes=[[1,0]], freq=1.0, amp=1.0, phase=0.0, inc=90, osParam= 2, observed = True)

class TestPulseyMethods(unittest.TestCase):

    def test_star(self, star):
        self.assertTrue(star)

    def test_flux(self, star):
        self.assertTrue(star.computeFlux(np.arange(0,1,0.05)))

    def test_binary(self, star):
        self.assertTrue(star.insertBinary(m1=1.0, r1=1.0, m2=1.0, r2=1.0, period=1.0, tTransit=0.0))
        self.assertTrue(star.binaryFlux())

    def test_show(self, star):
        self.assertTrue(star.show(0.0))

    def get_mock():
        """Returns a mock data table"""
        return Table({
            'EL': np.array([-15.0]), # Altitude (below horizon)
            'AZ': np.array([280.0]), # Azimuth
            'targetname': [''],
            'datetime_str': ['2025-Aug-05 10:00']
        })

if __name__ == '__main__':
    unittest.main()