"""Drop CUDA torch packages from a uv export so the ROCm build stays installed."""
import sys

skip_exact = {"torch", "torchaudio", "triton"}


def package_name(line: str) -> str:
    token = line.split("==", 1)[0].split("@", 1)[0].strip()
    return token


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    skipping = False
    kept = []
    with open(src, encoding="utf-8") as handle:
        for line in handle:
            is_header = (
                line[:1] not in ("", " ", "#", "-")
                and "==" in line
            )
            if is_header:
                name = package_name(line)
                skipping = name in skip_exact or name.startswith("nvidia-")
            if not skipping:
                kept.append(line)
    with open(dst, "w", encoding="utf-8") as handle:
        handle.writelines(kept)


if __name__ == "__main__":
    main()
