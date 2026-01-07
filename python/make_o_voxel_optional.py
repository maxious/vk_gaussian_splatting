#!/usr/bin/env python3
"""Script to make o_voxel imports optional in TRELLIS.2 vendored package."""

import re
from pathlib import Path

# Find all Python files in the vendored TRELLIS.2 package
trellis2_dir = Path(__file__).parent / "offline" / "vkgs_trellis2"
python_files = list(trellis2_dir.rglob("*.py"))

# Pattern to match lines with "import o_voxel" that are NOT already wrapped
# We want to match: "    import o_voxel" but NOT "try:\n    import o_voxel"
import_pattern = re.compile(r"^(\s+)import o_voxel$")

# Replacement: wrap in try/except with proper 4-space indentation
replacement = r"\1try:\n\1import o_voxel\n\1except ImportError:\n\1o_voxel = None"

count = 0
for py_file in python_files:
    # Skip __pycache__
    if "__pycache__" in str(py_file):
        continue

    try:
        with open(py_file, "r") as f:
            content = f.read()

        # Apply replacement - it will skip lines already wrapped (try: present)
        new_content, n = import_pattern.subn(replacement, content)

        if n > 0:
            with open(py_file, "w") as f:
                f.write(new_content)
            count += n
            print(f"Modified: {py_file.relative_to(trellis2_dir.parent.parent)}")

    except Exception as e:
        print(f"Error processing {py_file}: {e}")

print(f"\nTotal modifications: {count}")
