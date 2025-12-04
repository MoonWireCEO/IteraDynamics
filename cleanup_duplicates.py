import os
from pathlib import Path

# Define paths
core_path = Path("apex_core")
alpha_path = Path("alpha_engine")

print(f"🧹 Cleaning up duplicates in {alpha_path}...")

# Get list of files in core (The Source of Truth)
core_files = set(f.name for f in core_path.glob("*.py"))

deleted_count = 0

for fname in core_files:
    # Skip __init__.py (each folder needs its own)
    if fname == "__init__.py":
        continue
        
    dupe_file = alpha_path / fname
    
    if dupe_file.exists():
        try:
            os.remove(dupe_file)
            print(f"❌ Deleted duplicate: {fname}")
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Failed to delete {fname}: {e}")

print("-" * 40)
print(f"✅ Cleanup Complete. Deleted {deleted_count} files from alpha_engine.")