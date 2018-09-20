from tests import tests_perfomance
import logging

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    test = tests_perfomance.TestKMeansPerfomance()
    test.run_perfomance_test()
