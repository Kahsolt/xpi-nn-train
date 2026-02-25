#!/usr/bin/env python3
# Author: Armit
# Create Time: 周四 2026/02/26 

import os
from pathlib import Path

BASE_PATH = Path(__file__).parent

if 'DATA_ROOT' in os.environ:
  DATA_ROOT = Path.expanduser(os.environ['DATA_ROOT'])
else:
  DATA_ROOT = BASE_PATH / 'DATA_ROOT'
