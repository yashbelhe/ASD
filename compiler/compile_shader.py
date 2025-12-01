"""
Thin wrapper around `shader_transformer_tree_sitter.py` to compile Slang shaders.

Usage:
    from compiler.compile_shader import compile_if_needed
    compile_if_needed(Path("slang/circles.slang"))
"""

from pathlib import Path
import time
import hashlib

# Import transformer utilities without running its main
from . import shader_transformer_tree_sitter as transformer


def _read(path: Path) -> str:
    return path.read_text()


def compile_shader(src: Path, dst: Path) -> None:
    """Compile a single Slang shader from src -> dst."""
    source_code = _read(src)

    preprocessed = transformer.preprocess_source(source_code)
    source_bytes = preprocessed.encode("utf8")
    tree = transformer.parser.parse(source_bytes)
    transformed = transformer.transform_shader(source_bytes, tree)
    final = transformer.post_process_source(transformed.decode("utf8"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(final)


def _content_hash(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def compile_if_needed(src: Path, dst: Path) -> bool:
    """
    Compile if dst is missing or stale.
    Returns True if compilation happened, else False.
    """
    src = src.resolve()
    dst = dst.resolve()
    transformer_path = Path(transformer.__file__).resolve()

    if not dst.exists():
        compile_shader(src, dst)
        return True

    # Heuristic staleness: if src or transformer changed since dst write.
    dst_time = dst.stat().st_mtime
    if src.stat().st_mtime > dst_time or transformer_path.stat().st_mtime > dst_time:
        compile_shader(src, dst)
        return True

    # Hash check to catch same mtime but content changes (rare).
    src_hash = _content_hash(src)
    last_hash = getattr(compile_if_needed, "_last_src_hash", None)
    if last_hash != src_hash:
        compile_shader(src, dst)
        compile_if_needed._last_src_hash = src_hash
        return True

    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compile a Slang shader with __gen__ prefix.")
    parser.add_argument("src", type=Path, help="Source .slang file")
    parser.add_argument("--dst", type=Path, help="Destination .slang file (defaults to __gen__<name> next to src)")
    args = parser.parse_args()

    src = args.src
    dst = args.dst or src.with_name(f"__gen__{src.name}")
    did_compile = compile_if_needed(src, dst)
    print(f"{'Compiled' if did_compile else 'Up to date'}: {dst}")
