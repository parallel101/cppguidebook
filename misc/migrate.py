#!/usr/bin/env python3
"""One-time migration script: MkDocs → mdBook.

Transforms all src/*.md files in-place.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

REPO_URL = "https://github.com/parallel101/cppguidebook"

ICON_MAP = {
    "icon.fun": "note",
    "icon.tip": "tip",
    "icon.warn": "warning",
    "icon.story": "info",
    "icon.detail": "abstract",
    "icon2.fun": "note",
    "icon2.tip": "tip",
    "icon2.warn": "warning",
    "icon2.story": "info",
    "icon2.detail": "abstract",
}

# Regex for icon callout start: > {{ icon.xxx }} text...
ICON_RE = re.compile(r"^>\s*\{\{\s*(icon2?\.\w+)\s*\}\}\s?(.*)")

# Regex for image width attributes: ![alt](path){width=NNNpx} or {width="NNN"}
IMG_WIDTH_RE = re.compile(
    r'!\[([^\]]*)\]\(([^)]+)\)\{width="?(\d+)(?:px)?"?\}'
)

# Linked image with width: [![alt](path){width=NNNpx}](link)
LINKED_IMG_WIDTH_RE = re.compile(
    r'\[!\[([^\]]*)\]\(([^)]+)\)\{width="?(\d+)(?:px)?"?\}\]\(([^)]+)\)'
)

# Code fence language fixes
FENCE_FIXES = {
    "nasm": "x86asm",
    "c++": "cpp",
    "cuda": "cpp",
}
FENCE_RE = re.compile(r"^```(" + "|".join(re.escape(k) for k in FENCE_FIXES) + r")$")


def convert_callouts(lines: list[str]) -> list[str]:
    """Convert icon blockquote callouts to mdbook-admonish fenced blocks."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = ICON_RE.match(lines[i])
        if m:
            icon_key = m.group(1)
            first_text = m.group(2)
            admonish_type = ICON_MAP.get(icon_key, "note")

            # Collect all continuation lines of this blockquote
            body_lines = []
            if first_text:
                body_lines.append(first_text)

            j = i + 1
            while j < len(lines):
                line = lines[j]
                if line.startswith("> "):
                    body_lines.append(line[2:])
                    j += 1
                elif line.rstrip() == ">":
                    body_lines.append("")
                    j += 1
                else:
                    break
            i = j

            # Emit admonish block
            out.append(f"```admonish {admonish_type}")
            for bl in body_lines:
                out.append(bl)
            out.append("```")
        else:
            out.append(lines[i])
            i += 1
    return out


def process_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")

    result = []
    for line in lines:
        # Strip Jinja2 directives
        stripped = line.strip()
        if stripped in ("{% set icon = icon2 %}", "{% raw %}", "{% endraw %}"):
            continue

        # Remove [TOC]
        if stripped == "[TOC]":
            continue

        # Remove <!-- PGn --> comments
        if re.fullmatch(r"<!-- PG\d+ -->", stripped):
            continue

        # Replace template variables
        line = line.replace("{{ config.repo_url }}", REPO_URL)
        line = line.replace("{{ build_date }}", "BUILD_DATE_PLACEHOLDER")

        # Fix code fence languages
        fm = FENCE_RE.match(line)
        if fm:
            line = "```" + FENCE_FIXES[fm.group(1)]

        # Convert linked images with width (must be before plain image width)
        line = LINKED_IMG_WIDTH_RE.sub(
            r'<a href="\4"><img src="\2" alt="\1" width="\3"></a>', line
        )

        # Convert image width attributes to HTML
        line = IMG_WIDTH_RE.sub(r'<img src="\2" alt="\1" width="\3">', line)

        result.append(line)

    # Convert icon callouts
    result = convert_callouts(result)

    new_text = "\n".join(result)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"  migrated: {path.name}")
    else:
        print(f"  unchanged: {path.name}")


def main() -> None:
    md_files = sorted(SRC.glob("*.md"))
    print(f"Processing {len(md_files)} files in {SRC}")
    for f in md_files:
        process_file(f)
    print("Done!")


if __name__ == "__main__":
    main()
