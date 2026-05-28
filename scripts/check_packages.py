import importlib
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError

packages = {
    "faiss": "faiss-cpu",
    "langchain": "langchain",
    "mistralai": "mistralai",
}

all_ok = True
for module, pip_name in packages.items():
    try:
        importlib.import_module(module)
        v = version(pip_name)
        print(f"[OK] {pip_name} {v}")
    except ImportError:
        print(f"[FAIL] {pip_name} not importable — run: pip install {pip_name}")
        all_ok = False
    except PackageNotFoundError:
        print(f"[WARN] {pip_name} imported but version not found in metadata")

print()
result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("[OK] No dependency conflicts detected.")
else:
    print("[FAIL] Dependency conflicts found:")
    print(result.stdout.strip())
    all_ok = False

if not all_ok:
    sys.exit(1)
