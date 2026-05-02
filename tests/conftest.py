import os
import sys

# Ensure project root is on the path regardless of how pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
