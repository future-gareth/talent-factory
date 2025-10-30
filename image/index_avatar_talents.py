#!/usr/bin/env python3
"""
Entry point for indexing avatar talents.
Can be run as: python -m talent_factory.image.index_avatar_talents
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from image.indexer import main

if __name__ == "__main__":
    main()
