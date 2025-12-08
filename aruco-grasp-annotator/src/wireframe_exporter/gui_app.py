#!/usr/bin/env python3
"""
GUI Application launcher for Wireframe Exporter.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from gui.main_window import main

if __name__ == "__main__":
    main()
