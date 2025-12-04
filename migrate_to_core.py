import os
import shutil
import difflib
from pathlib import Path

# The threshold for "Identical"
THRESHOLD = 85.0

def get_python_files(folder):
    files = {}
    for r, d, f in os.walk(folder):
        for file in f:
            if file.endswith(".py") and "__init__" not in file:
                path = Path(r) / file
                try:
                    files[file] = path.read_text(encoding='utf-8')
                except:
                    pass
    return files

def migrate():
    print("🚀 STARTING MIGRATION: MoonWire/AlphaEngine -> Apex Core")
    
    crypto_files = get_python_files("moonwire")
    equity_files = get_python_files("alpha_engine")
    
    # Ensure destination exists
    core_path = Path("apex_core")
    core_path.mkdir(exist_ok=True)
    
    common_names = set(crypto_files.keys()) & set(equity_files.keys())
    
    moved_count = 0
    
    for fname in common_names:
        code_a = crypto_files[fname]
        code_b = equity_files[fname]
        
        seq = difflib.SequenceMatcher(None, code_a, code_b)
        ratio = seq.ratio() * 100
        
        if ratio > THRESHOLD:
            print(f"📦 MOVING: {fname} ({ratio:.1f}% match)")
            
            # 1. Move from MoonWire to Apex Core
            src = Path("moonwire") / fname
            dst = core_path / fname
            
            if src.exists():
                shutil.move(str(src), str(dst))
            
            # 2. Delete from AlphaEngine (since it's a duplicate)
            dupe = Path("alpha_engine") / fname
            if dupe.exists():
                os.remove(dupe)
                
            moved_count += 1
            
    print("-" * 50)
    print(f"✅ MIGRATION COMPLETE. {moved_count} files moved to apex_core.")
    print("⚠️  NEXT STEP: You must update imports in your remaining files!")
    print("    Example: Change 'from .utils import X' to 'from apex_core.utils import X'")

if __name__ == "__main__":
    migrate()