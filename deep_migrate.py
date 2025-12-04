import os
import shutil
import difflib
from pathlib import Path

# Config
THRESHOLD = 85.0
CORE_DIR = Path("apex_core")
CRYPTO_DIR = Path("moonwire")
EQUITY_DIR = Path("alpha_engine")

def get_file_map(folder):
    """Returns a dict mapping {filename: full_path}"""
    file_map = {}
    # rglog('*') finds all files recursively
    for path in folder.rglob("*.py"):
        if "__init__" not in path.name:
            file_map[path.name] = path
    return file_map

def migrate():
    print("🚀 STARTING DEEP MIGRATION...")
    
    # 1. Map all files in both repos (finding them wherever they hide)
    crypto_map = get_file_map(CRYPTO_DIR)
    equity_map = get_file_map(EQUITY_DIR)
    
    CORE_DIR.mkdir(exist_ok=True)
    
    # Find common filenames
    common_names = set(crypto_map.keys()) & set(equity_map.keys())
    
    moved_count = 0
    
    for fname in common_names:
        src_path = crypto_map[fname]
        dupe_path = equity_map[fname]
        
        # Read content
        try:
            code_a = src_path.read_text(encoding='utf-8')
            code_b = dupe_path.read_text(encoding='utf-8')
        except:
            print(f"⚠️ Could not read {fname}, skipping.")
            continue

        # Check similarity
        seq = difflib.SequenceMatcher(None, code_a, code_b)
        ratio = seq.ratio() * 100
        
        if ratio > THRESHOLD:
            print(f"📦 MOVING: {fname} (found in {src_path.parent})")
            
            # 1. Move to Apex Core root (Flattening)
            dst = CORE_DIR / fname
            
            # Don't overwrite if it already exists in core
            if not dst.exists():
                shutil.move(str(src_path), str(dst))
                moved_count += 1
            else:
                print(f"   Skipping move (File already in core): {fname}")
                # If it's already in core, we can safely delete the source
                os.remove(src_path)

            # 2. Delete the duplicate in Alpha Engine
            if dupe_path.exists():
                os.remove(dupe_path)
                print(f"   ❌ Deleted duplicate in alpha_engine")

    print("-" * 50)
    print(f"✅ DEEP MIGRATION COMPLETE. {moved_count} files moved to apex_core.")

if __name__ == "__main__":
    migrate()