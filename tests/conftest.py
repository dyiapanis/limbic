"""Make `import limbic` resolve to THIS repo, not any installed/live copy."""
import os
import sys

# The repo root IS the `limbic` package — put its PARENT on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))