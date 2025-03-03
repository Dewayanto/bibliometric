import unittest
import sys

# Import test modules
from tests.test_data_loader import TestDataLoader
from tests.test_analyzer import TestBibliometricAnalyzer
from tests.test_network_analysis import TestNetworkAnalyzer

def run_tests():
    """Run all unit tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestBibliometricAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkAnalyzer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return 0 if successful, 1 if there were failures
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())
