"""Python port of the shop's setSvgImgColor / setSvgImgRoseVisibility (reference/shop-configurator.js)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
SHOP_CLASSES = ["st0", "st1", "st2", "st3", "st5"]
COLOR_CLASS_RE = re.compile(r"st\d_\d+(_pattern)?")
USED_RE = re.compile(r"^st\d_(\d+)(_pattern)?$")


@dataclass
class SimConfig:
    colors: dict[int, str]  # color index -> css color or "url(...)"
    pattern_index: int | None = None
    rose: bool = True


@dataclass
class SimResult:
    visible_color_elements: list[str] = field(default_factory=list)
    unfilled: list[str] = field(default_factory=list)
    svg: str = ""


def _classes(e) -> list[str]:
    return (e.get("class") or "").split()


def _get_style(e, prop: str) -> str | None:
    for part in (e.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            if k.strip() == prop:
                return v.strip()
    return None


def _set_style(e, prop: str, value: str) -> None:
    parts = [p for p in (e.get("style") or "").split(";")
             if p.strip() and p.split(":", 1)[0].strip() != prop]
    parts.append(f"{prop}:{value}")
    e.set("style", ";".join(parts))


def _by_class(root, cls: str):
    return [e for e in root.iter() if isinstance(e.tag, str) and cls in _classes(e)]


def _by_id(root, i: str):
    for e in root.iter():
        if isinstance(e.tag, str) and e.get("id") == i:
            return e
    return None


def _visible(e) -> bool:
    while e is not None:
        if _get_style(e, "display") == "none":
            return False
        e = e.getparent()
    return True


def _set_svg_img_color(root, color_index: int, value: str, state: dict) -> None:
    if not color_index:  # JS: `!colorIndex` returns early
        return
    image = value[4:-1].replace('"', "").replace("'", "") if value.startswith("url(") else ""
    if image:
        state["show_pattern"] = True
        state["pattern_index"] = color_index
    elif state["pattern_index"] == color_index:
        state["show_pattern"] = False
    for cn in SHOP_CLASSES:
        elements = _by_class(root, f"{cn}_{color_index}{'_pattern' if image else ''}")
        if not elements:
            continue
        if image:
            pat = _by_id(root, f"image_{cn}_{color_index}_pattern")
            img = next(pat.iter(f"{{{SVG}}}image"), None) if pat is not None else None
            if img is not None:  # JS: `if (imageElement)` guard
                pat.set("width", "80")
                pat.set("height", "80")
                img.set("width", "80")
                img.set("height", "80")
                img.set(f"{{{XLINK}}}href", image)
                for e in elements:
                    _set_style(e, "fill", f"url(#image_{cn}_{color_index}_pattern)")
        else:
            for e in elements:
                _set_style(e, "fill", value)
    pl, sl = _by_id(root, "Layer_3"), _by_id(root, "Layer_2_1")
    if pl is not None and sl is not None:
        _set_style(pl, "display", "inline" if state["show_pattern"] else "none")
        _set_style(sl, "display", "none" if state["show_pattern"] else "inline")


def _set_rose(root, show: bool) -> None:
    for e in _by_class(root, "st4"):
        _set_style(e, "display", "inline" if show else "none")
    for e in _by_class(root, "st_circel"):
        _set_style(e, "display", "none" if show else "inline")


def simulate(svg: str, cfg: SimConfig) -> SimResult:
    root = etree.fromstring(svg.encode("utf-8"))
    tree = root.getroottree()
    state = {"show_pattern": False, "pattern_index": None}
    for idx, val in cfg.colors.items():
        _set_svg_img_color(root, int(idx), val, state)
    _set_rose(root, cfg.rose)

    res = SimResult()
    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        if not any(COLOR_CLASS_RE.fullmatch(c) for c in _classes(e)) or not _visible(e):
            continue
        path = tree.getpath(e)
        res.visible_color_elements.append(path)
        if not _get_style(e, "fill"):
            res.unfilled.append(path)
    res.svg = etree.tostring(root, encoding="unicode")
    return res


def sim_issues(svg: str) -> list[str]:
    """One description per visible color element the shop logic leaves unfilled."""
    try:
        root = etree.fromstring(svg.encode("utf-8"))
    except etree.XMLSyntaxError:
        return []
    tree = root.getroottree()
    used: set[int] = set()
    has_pattern = False
    for e in root.iter():
        if not isinstance(e.tag, str):
            continue
        for c in _classes(e):
            m = USED_RE.match(c)
            if m:
                used.add(int(m.group(1)))
                has_pattern = has_pattern or c.endswith("_2_pattern")
    if not used:
        return []
    base = {i: f"#{(i * 37) % 256:02x}{(i * 71 + 50) % 256:02x}{(i * 113 + 90) % 256:02x}"
            for i in sorted(used)}
    configs = [(None, base)]
    if has_pattern:
        configs.append((2, {**base, 2: "url(x.png)"}))
    out: list[str] = []
    seen: set[str] = set()
    for pidx, colors in configs:
        for rose in (True, False):
            res = simulate(svg, SimConfig(colors=colors, pattern_index=pidx, rose=rose))
            for path in res.unfilled:
                el = tree.xpath(path)[0]
                desc = f'<{etree.QName(el).localname} class="{el.get("class")}"> at {path}'
                if desc not in seen:
                    seen.add(desc)
                    out.append(desc)
    return out
