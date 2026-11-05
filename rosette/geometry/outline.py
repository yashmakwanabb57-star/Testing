from __future__ import annotations

from ..shapes import Shape


def outlines(shapes: list[Shape]) -> list[Shape]:
    """One st6 outline per input shape (the writer applies st6 for kind=outline)."""
    out: list[Shape] = []
    for s in shapes:
        if s.tag == "polygon":
            nums = s.attrs["points"].split()
            attrs = {"points": " ".join(nums + nums[:1])}
            out.append(Shape("polyline", attrs, None, role=s.role, z=s.z, kind="outline"))
        else:
            attrs = {k: v for k, v in s.attrs.items() if k != "fill-rule"}
            out.append(Shape(s.tag, attrs, None, role=s.role, z=s.z, kind="outline"))
    return out
