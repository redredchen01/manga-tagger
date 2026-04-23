"""Build the allowed-tag-list prompt fragment from the library.

Used by VLM prompts to constrain output to known canonical tags.
The library is grouped by category; the 'other' bucket is omitted because
those entries lack a meaningful category and would only add noise.
"""
from __future__ import annotations

from typing import Dict, List

# Display order — character first because it's most salient for tagging
CATEGORY_ORDER = ["character", "clothing", "body", "action", "theme", "style"]

CATEGORY_LABEL_ZH = {
    "character": "角色 (Character)",
    "clothing": "服裝 (Clothing)",
    "body": "身體特徵 (Body)",
    "action": "動作與互動 (Action)",
    "theme": "主題 (Theme)",
    "style": "藝術風格 (Style)",
}


def group_by_category(library: List[Dict]) -> Dict[str, List[Dict]]:
    """Group library entries by their category field."""
    grouped: Dict[str, List[Dict]] = {}
    for entry in library:
        cat = entry.get("category", "other")
        grouped.setdefault(cat, []).append(entry)
    return grouped


def build_prompt_fragment(library: List[Dict]) -> str:
    """Build a prompt fragment listing all allowed tags grouped by category.

    Format:
        ### 角色 (Character)
        - 貓娘：貓耳貓尾
        - 狐娘：狐耳狐尾

        ### 服裝 (Clothing)
        - 和服：日式
        ...
    """
    grouped = group_by_category(library)
    out_lines: List[str] = []
    for cat in CATEGORY_ORDER:
        entries = grouped.get(cat, [])
        if not entries:
            continue
        out_lines.append(f"### {CATEGORY_LABEL_ZH[cat]}")
        for e in entries:
            name = e.get("tag_name", "").strip()
            desc = (e.get("description", "") or "").strip().replace("\n", " ")
            if not name:
                continue
            if desc:
                # Truncate overly long descriptions to keep prompt size sane
                if len(desc) > 60:
                    desc = desc[:60] + "…"
                out_lines.append(f"- {name}：{desc}")
            else:
                out_lines.append(f"- {name}")
        out_lines.append("")  # blank line between sections
    return "\n".join(out_lines).strip()


def build_compact_prompt_fragment(library: List[Dict]) -> str:
    """Build a compact allowed-tag fragment: tag names only, grouped by category.

    Format:
        ### 角色
        蘿莉, 貓娘, 狐娘, ...

        ### 服裝
        女生制服, 男生制服, ...

    For the 611-tag library this emits ~3,500 chars (vs ~14,675 for the
    verbose build_prompt_fragment). The verbose descriptions drop out;
    the VLM's pretrained knowledge covers semantics, only tag names
    matter for library matching.
    """
    grouped = group_by_category(library)
    out_lines: List[str] = []
    for cat in CATEGORY_ORDER:
        entries = grouped.get(cat, [])
        if not entries:
            continue
        # Use the Chinese portion of the category label only — drop the
        # parenthesised English to save tokens.
        label_zh = CATEGORY_LABEL_ZH[cat].split(" (")[0]
        out_lines.append(f"### {label_zh}")
        names = []
        seen_names = set()
        for e in entries:
            name = (e.get("tag_name") or "").strip()
            if not name or name in seen_names:
                continue
            names.append(name)
            seen_names.add(name)
        out_lines.append(", ".join(names))
        out_lines.append("")  # blank line between sections
    return "\n".join(out_lines).strip()
