#!/usr/bin/env python3
"""
Simple runner script for the Symmetry Exporter
"""

import sys
import os
from pathlib import Path

# Add the symmetry_exporter directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app import main

if __name__ == "__main__":
    main()

