"""Merge report/sections/q*.md into report/report.md."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTIONS_DIR = REPO_ROOT / "report" / "sections"
OUT_PATH = REPO_ROOT / "report" / "report.md"

GITHUB_REPO_URL = "https://github.com/ttqureshi/DL4STG-PA3"

SECTION_ORDER = (
    "q1.md",
    "q2.md",
    "q3.md",
    "q4.md",
    "q5.md",
    "q6.md",
    "q7.md",
)


def main() -> None:
    parts = [
        "# DL4STG — Programming Assignment 3\n",
        "\n",
        GITHUB_PLACEHOLDER,
        "---\n\n",
    ]
    for name in SECTION_ORDER:
        path = SECTIONS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing section {path}")
        parts.append(path.read_text(encoding="utf-8").rstrip())
        parts.append("\n\n---\n\n")
    OUT_PATH.write_text("".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()