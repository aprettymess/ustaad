"""Synthetic 'tyre change' videos with solid-colour parts, for testing the pipeline with --mock.

  python scripts/make_synthetic.py out/            -> master.mp4, attempt_order.mp4, attempt_ok.mp4

Colours match perception/sam.py MockTracker.COLORS.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 400, 25
WHEEL, NUT, JACK = (200, 60, 30), (0, 220, 255), (40, 40, 220)
HAND, FACE, PLATE = (120, 170, 230), (180, 120, 255), (245, 245, 245)

NUTS_ON = [(150, 170), (210, 170), (150, 230), (210, 230)]
NUTS_TRAY = [(470, 330), (500, 330), (530, 330), (560, 330)]
JACK_FROM, JACK_TO = (520, 200), (180, 330)
WHEEL_ON, WHEEL_OFF = (180, 200), (-120, 200)


def lerp(a, b, u):
    return (int(a[0] + (b[0] - a[0]) * u), int(a[1] + (b[1] - a[1]) * u))


def draw(state, hand=None):
    img = np.full((H, W, 3), 128, np.uint8)
    cv2.rectangle(img, (20, 20), (120, 50), PLATE, -1)                  # licence plate
    cv2.putText(img, "LEA-123", (26, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    cv2.circle(img, (590, 50), 28, FACE, -1)                             # bystander face
    wx, wy = state["wheel"]
    cv2.rectangle(img, (wx - 80, wy - 80), (wx + 80, wy + 80), WHEEL, -1)
    for (x, y) in state["nuts"]:
        cv2.rectangle(img, (x - 9, y - 9), (x + 9, y + 9), NUT, -1)
    jx, jy = state["jack"]
    cv2.rectangle(img, (jx - 45, jy - 18), (jx + 45, jy + 18), JACK, -1)
    if hand:
        cv2.ellipse(img, hand, (30, 22), 0, 0, 360, HAND, -1)
    return img


def make(path, order):
    st = {"wheel": WHEEL_ON, "nuts": list(NUTS_ON), "jack": JACK_FROM}
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    def hold(sec):
        for _ in range(int(sec * FPS)):
            vw.write(draw(st))

    def anim(sec, update):
        n = int(sec * FPS)
        for i in range(n):
            u = (i + 1) / n
            hand = update(u)
            vw.write(draw(st, hand))

    def nuts(u):
        st["nuts"] = [lerp(a, b, u) for a, b in zip(NUTS_ON, NUTS_TRAY)]
        return lerp(NUTS_ON[0], NUTS_TRAY[0], u)

    def jack(u):
        st["jack"] = lerp(JACK_FROM, JACK_TO, u)
        return (st["jack"][0], st["jack"][1] - 40)

    def wheel(u):
        st["wheel"] = lerp(WHEEL_ON, WHEEL_OFF, u)
        st["nuts"] = list(st["nuts"])
        return (st["wheel"][0] + 90, st["wheel"][1])

    acts = {"nuts": nuts, "jack": jack, "wheel": wheel}
    hold(1.5)
    for a in order:
        anim(1.5, acts[a])
        hold(1.5)
    vw.release()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
    out.mkdir(parents=True, exist_ok=True)
    make(out / "master.mp4", ["nuts", "jack", "wheel"])
    make(out / "attempt_order.mp4", ["jack", "nuts", "wheel"])   # jacked before loosening nuts
    make(out / "attempt_ok.mp4", ["nuts", "jack", "wheel"])
    print("wrote", *sorted(p.name for p in out.glob("*.mp4")))
