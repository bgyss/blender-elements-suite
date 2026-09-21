"""Build dist/blender_elements-<version>.zip and verify its layout.

Blender extracts the archive root directly into
extensions/<repo>/<id>/, so __init__.py and blender_manifest.toml must sit at
the root with nothing nested under a package directory. A nested layout makes
Python treat the installed directory as a namespace package and the extension
fails to load.
"""

import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "addon" / "blender_elements"
VERSION = "0.1.0"
OUT = ROOT / "dist" / f"blender_elements-{VERSION}.zip"

REQUIRED_ROOT_FILES = {"__init__.py", "blender_manifest.toml", "LICENSE"}


def build() -> pathlib.Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            zf.write(path, path.relative_to(SRC))
    return OUT


def verify(path: pathlib.Path) -> None:
    names = set(zipfile.ZipFile(path).namelist())
    missing = REQUIRED_ROOT_FILES - names
    if missing:
        raise SystemExit(f"archive root is missing {sorted(missing)}; got {sorted(names)}")
    nested = [n for n in names if n.startswith("blender_elements/")]
    if nested:
        raise SystemExit(f"package contents must not be nested: {nested}")
    print(f"built {path} with {len(names)} files")


if __name__ == "__main__":
    verify(build())
    sys.exit(0)
