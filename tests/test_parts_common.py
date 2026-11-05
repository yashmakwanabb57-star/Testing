"""blob_rings: rings of blobs around the head center (star tips, loop ends)."""
import math

import cv2
import numpy as np

from rosette.vision.parts_common import blob_rings

HEAD = (500.0, 500.0, 300.0)


def _blobs(angles_deg, r=180.0, size=10):
    img = np.zeros((1000, 1000), np.uint8)
    for a in angles_deg:
        x = HEAD[0] + r * math.cos(math.radians(a))
        y = HEAD[1] + r * math.sin(math.radians(a))
        cv2.circle(img, (int(round(x)), int(round(y))), size, 1, -1)
    return img > 0


def test_six_evenly_spread_blobs_are_a_ring():
    rings = blob_rings(_blobs([10 + 60 * k for k in range(6)]), HEAD, 1.0)
    assert len(rings) == 1
    assert rings[0].visible == 6 and rings[0].order == 6
    assert abs(rings[0].rotation_deg - 10) < 1


def test_blobs_bunched_on_one_side_are_not_a_ring():
    # 4 blobs within 24 degrees: |mean exp(i 4 theta)| is 0.82, above RING_EVEN,
    # although they cover a fifteenth of the circle
    assert blob_rings(_blobs([0, 8, 16, 24]), HEAD, 1.0) == []
