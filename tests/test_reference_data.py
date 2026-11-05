from pathlib import Path

from lxml import etree

REF = Path(__file__).resolve().parent.parent / "reference"
NS = {"s": "http://www.w3.org/2000/svg"}


def _canonical_ids():
    return [l for l in (REF / "canonical.txt").read_text().splitlines() if l.strip()]


def test_reference_counts():
    assert len(list((REF / "svg").glob("*.svg"))) == 79
    assert len(list((REF / "photos").glob("*.png"))) == 76
    assert len(_canonical_ids()) == 35


def test_canonical_have_toggle_groups():
    for cid in _canonical_ids():
        root = etree.parse(str(REF / "svg" / f"{cid}.svg")).getroot()
        for gid in ("Layer_3", "Layer_2_1"):
            assert root.xpath(f"//*[@id='{gid}']"), f"{cid} missing {gid}"


def test_rose_asset():
    root = etree.parse(str(REF / "rose.svg")).getroot()
    paths = root.findall(".//s:path", NS)
    lines = root.findall(".//s:polyline", NS)
    assert len(paths) == 1 and len(lines) == 2
    assert all(e.get("class") == "st4" for e in paths + lines)
