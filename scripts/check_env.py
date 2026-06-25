"""环境自检脚本：在项目根目录、已激活 .venv 后运行 python scripts/check_env.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("Python:", sys.version.split()[0])
    print()

    errors = []

    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
    except ImportError as e:
        errors.append(f"torch: {e}")

    for pkg in ("web3", "numpy", "matplotlib", "pandas", "dotenv", "pytest"):
        try:
            mod = __import__(pkg if pkg != "dotenv" else "dotenv")
            ver = getattr(mod, "__version__", "ok")
            print(f"{pkg}: {ver}")
        except ImportError as e:
            errors.append(f"{pkg}: {e}")

    print()
    try:
        from config.settings import RPC, PSO_CONFIG

        print("config/settings.py: ok")
        print("  DEVICE =", PSO_CONFIG["device"])
        print("  ETH HTTP configured:", bool(RPC["ethereum"]["http"]))
    except Exception as e:
        errors.append(f"config: {e}")

    print()
    if errors:
        print("FAILED:")
        for err in errors:
            print(" -", err)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
