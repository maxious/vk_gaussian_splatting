#!/usr/bin/env python3
"""
Patch torch_python.dll to fix Windows 'int too large to convert to C long' error
when using torch.compile with mode='reduce-overhead' on Windows.

This script patches the format specifier in the StaticCudaLauncher from %l (32-bit long)
to use a 64-capable format.

BEFORE RUNNING: Ensure no Python processes are using torch (close all terminals, IDEs, etc.)
"""

import os
import shutil
import struct
import sys
from pathlib import Path


def find_torch_python_dll(venv_path: Path | None = None) -> Path | None:
    """Find torch_python.dll in the virtual environment."""
    if venv_path:
        lib_path = venv_path / "Lib" / "site-packages" / "torch" / "lib"
        if lib_path.exists():
            return lib_path / "torch_python.dll"

    try:
        import torch

        if torch.__file__ is None:
            return None
        torch_root = Path(torch.__file__).parent
        lib_path = torch_root / "lib"
        dll_path = lib_path / "torch_python.dll"

        if dll_path.exists():
            return dll_path
    except (ImportError, TypeError):
        pass

    return None


def check_dll_in_use(dll_path: Path) -> bool:
    """Check if any process is currently using the DLL."""
    if sys.platform != "win32":
        print(f"Warning: DLL usage check only works on Windows. Skipping check.")
        return False

    import subprocess

    try:
        result = subprocess.run(
            ["tasklist", "/m", str(dll_path)], capture_output=True, text=True, check=True
        )
        if result.stdout and "torch_python.dll" in result.stdout:
            lines = [line for line in result.stdout.split("\n") if line.strip()]
            if len(lines) > 1:
                return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Warning: Could not check if DLL is in use.")

    return False


def patch_dll(dll_path: Path, backup: bool = True) -> bool:
    """
    Patch torch_python.dll to fix the overflow error.

    Searches for 'KiiiiisOl' and replaces the final 'l' (0x6C) with 'K' (0x4B).
    """
    dll_path = Path(dll_path).resolve()

    if not dll_path.exists():
        print(f"Error: DLL not found at {dll_path}")
        return False

    if check_dll_in_use(dll_path):
        print(f"Error: DLL is currently in use by another process.")
        print(f"Please close all Python terminals and IDEs, then run this script again.")
        return False

    if backup:
        backup_path = dll_path.with_suffix(".dll.backup")
        try:
            shutil.copy2(dll_path, backup_path)
            print(f"Backup created at: {backup_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False

    try:
        with open(dll_path, "rb") as f:
            dll_data = bytearray(f.read())
    except Exception as e:
        print(f"Error reading DLL: {e}")
        return False

    search_pattern = b"KiiiiisOl"

    print(f"Searching for pattern: {search_pattern.decode('ascii')}")

    occurrences = []
    offset = 0
    while True:
        idx = dll_data.find(search_pattern, offset)
        if idx == -1:
            break
        occurrences.append(idx)
        offset = idx + 1

    if not occurrences:
        print(f"Error: Could not find pattern '{search_pattern.decode('ascii')}' in DLL.")
        print(f"This DLL may already be patched, or the torch version has changed.")
        return False

    print(f"Found {len(occurrences)} occurrence(s) of the pattern at offsets: {occurrences}")

    if len(occurrences) > 1:
        print(f"Warning: Found multiple occurrences. This is unusual.")
        print(f"Please verify these are correct before proceeding.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            return False

    patch_byte = ord("l")  # 0x6C
    new_byte = ord("K")  # 0x4B

    for offset in occurrences:
        # The pattern length is 9, so the last character is at offset + 8
        last_char_offset = offset + 8

        if dll_data[last_char_offset] != patch_byte:
            print(
                f"Warning: At offset {last_char_offset}, expected {hex(patch_byte)}, found {hex(dll_data[last_char_offset])}"
            )
            print(f"Skipping this occurrence.")
            continue

        print(f"Patching at offset {last_char_offset}: 0x{patch_byte:02X} -> 0x{new_byte:02X}")
        dll_data[last_char_offset] = new_byte

    try:
        with open(dll_path, "wb") as f:
            f.write(dll_data)
        print(f"Successfully patched DLL at: {dll_path}")
        return True
    except Exception as e:
        print(f"Error writing patched DLL: {e}")
        if backup:
            try:
                shutil.copy2(backup_path, dll_path)
                print("Restored DLL from backup.")
            except:
                pass
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Patch torch_python.dll to fix torch.compile overflow error on Windows"
    )
    parser.add_argument(
        "--dll-path", type=Path, help="Path to torch_python.dll (auto-detected if not specified)"
    )
    parser.add_argument(
        "--venv-path", type=Path, help="Path to virtual environment (for auto-detection)"
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Skip creating backup (not recommended)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only check if DLL can be patched, don't modify it"
    )

    args = parser.parse_args()

    if sys.platform != "win32":
        print("This script is designed for Windows only.")
        print("The overflow issue is Windows-specific due to 32-bit long type.")
        return 1

    print("=" * 70)
    print("torch_python.dll Patcher for Windows torch.compile Fix")
    print("=" * 70)
    print()

    if args.dll_path:
        dll_path = args.dll_path
    else:
        print("Searching for torch_python.dll...")
        dll_path = find_torch_python_dll(args.venv_path)

    if not dll_path:
        print("Error: Could not find torch_python.dll")
        print()
        print("Please run this script from your virtual environment or specify the path:")
        print()
        print("  # Activate your venv first")
        print("  uv run python scripts/patch_torch_dll.py")
        print()
        print("  # Or specify DLL path directly")
        print(
            '  python scripts/patch_torch_dll.py --dll-path "...\\Lib\\site-packages\\torch\\lib\\torch_python.dll"'
        )
        print()
        print("Typical location:")
        print("  <venv>\\Lib\\site-packages\\torch\\lib\\torch_python.dll")
        return 1

    print(f"Found DLL: {dll_path}")
    print()

    if args.dry_run:
        print("DRY RUN MODE - Will not modify files")
        if check_dll_in_use(dll_path):
            print("Status: DLL is in use by another process")
            return 1
        else:
            print("Status: DLL can be patched (not in use)")
            return 0

    print("This will modify torch_python.dll to fix the torch.compile overflow error.")
    print("The change:")
    print("  - Find pattern: 'KiiiiisOl'")
    print("  - Replace final 'l' (0x6C) with 'K' (0x4B)")
    print()

    if not args.no_backup:
        print(f"A backup will be created at: {dll_path.with_suffix('.dll.backup')}")

    response = input("Proceed with patching? (y/N): ")
    if response.lower() != "y":
        print("Aborted.")
        return 0

    print()
    print("Patching...")

    success = patch_dll(dll_path, backup=not args.no_backup)

    if success:
        print()
        print("=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print()
        print("The DLL has been patched.")
        print("Next time you run torch, the overflow error should be fixed.")
        print()
        print("If you encounter any issues, you can restore from the backup file.")
        return 0
    else:
        print()
        print("=" * 70)
        print("FAILED")
        print("=" * 70)
        print()
        print("The patching process failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
