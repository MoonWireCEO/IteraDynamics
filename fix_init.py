import os
from pathlib import Path

# The three packages we need to fix
packages = ["apex_core", "moonwire", "alpha_engine"]

print(f"Current Directory: {os.getcwd()}")
print("-" * 30)

for pkg in packages:
    # Define the correct path
    correct_file = Path(pkg) / "__init__.py"
    # Define the "Bad" path (the hidden text file)
    bad_file = Path(pkg) / "__init__.py.txt"
    
    # 1. Check if the bad file exists and rename it
    if bad_file.exists():
        print(f"FOUND BAD FILE: {bad_file}")
        bad_file.rename(correct_file)
        print(f" -> FIXED: Renamed to {correct_file}")
    
    # 2. Check if the correct file exists
    if correct_file.exists():
        print(f"CONFIRMED: {correct_file} exists.")
    else:
        # 3. If neither existed, create the correct one fresh
        print(f"MISSING: {correct_file} ... Creating it now.")
        correct_file.touch()
        print(f" -> CREATED: {correct_file}")

print("-" * 30)