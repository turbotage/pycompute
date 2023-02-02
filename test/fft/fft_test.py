import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from testrun import test_runner
test_runner()

from pycompute.cuda import cuda_program as cuda_cp




