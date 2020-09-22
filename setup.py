import glob
import os
import sys
from distutils.core import setup

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "python")))

setup(
    name="MCP-FS",
    description="MCP-FS",
    author="Us",
    version="0.1.0",
    packages=["mcp"],
    scripts=glob.glob("bin/*"),
    test_suite="tests",
)
