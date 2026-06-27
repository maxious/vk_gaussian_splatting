"""
Generate synthetic test images of a textured 3D box from two viewpoints.

Output:
  box_00.png  — view 1 (from left and slightly above)
  box_01.png  — view 2 (rotated ~8 deg, different elevation)

Uses only PIL (no GPU / torch).  Deterministic with random.seed(42).
"""

import math
import random
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Seeded RNG for reproducible output
# ---------------------------------------------------------------------------
random.seed(42)

IMG_SIZE = 512

# ---------------------------------------------------------------------------
# 3D cube geometry  (side length = 2, centred at origin)
# ---------------------------------------------------------------------------
# Vertex order:  see diagram below
#
#       3 ────── 2
#      /|       /|
#     / |      / |
#    0 ────── 1  |
#    |  7 ────|── 6
#    | /      | /
#    4 ────── 5
#
#   Front = 4-5-6-7   (green)
#   Back  = 0-1-2-3   (red)
#   Left  = 3-0-4-7   (purple)
#   Right = 1-2-6-5   (orange)
#   Top   = 2-3-7-6   (yellow)
#   Bottom = 0-1-5-4  (blue)

CUBE = [
    (-1, -1, -1),  # 0  back-left-bottom
    ( 1, -1, -1),  # 1  back-right-bottom
    ( 1,  1, -1),  # 2  back-right-top
    (-1,  1, -1),  # 3  back-left-top
    (-1, -1,  1),  # 4  front-left-bottom
    ( 1, -1,  1),  # 5  front-right-bottom
    ( 1,  1,  1),  # 6  front-right-top
    (-1,  1,  1),  # 7  front-left-top
]

# Each face: (vertex_indices, checker_color_A, checker_color_B, label)
FACES = [
    ((0, 1, 2, 3), (220, 60, 60),   (180, 40, 40),   "Back"),
    ((4, 5, 6, 7), (60, 220, 60),   (40, 180, 40),   "Front"),
    ((0, 1, 5, 4), (60, 60, 220),   (40, 40, 180),   "Bottom"),
    ((2, 3, 7, 6), (220, 220, 60),  (180, 180, 40),  "Top"),
    ((1, 2, 6, 5), (220, 120, 40),  (180, 90, 20),   "Right"),
    ((3, 0, 4, 7), (200, 60, 200),  (160, 40, 160),  "Left"),
]

CHECKER_N = 6  # subdivisions per face edge (6x6 grid)


# ---------------------------------------------------------------------------
# 3D math helpers
# ---------------------------------------------------------------------------
def rot_x(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c)


def rot_y(p, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)


def project(p, scale=160, cx=IMG_SIZE // 2, cy=IMG_SIZE // 2):
    """Orthographic-like projection with a small perspective tweak."""
    x, y, z = p
    d = 3.5  # viewer distance along z
    f = scale / (d + z)
    return (int(x * f + cx), int(-y * f + cy))  # flip y for screen


# ---------------------------------------------------------------------------
# Bilinear interpolation on a 3D quad
# ---------------------------------------------------------------------------
def lerp(a, b, t):
    return (a[i] + (b[i] - a[i]) * t for i in range(3))


def bilerp(v0, v1, v2, v3, u, v):
    """Bilinear point on quad (v0-v1-v2-v3) at parametric (u,v)."""
    top = tuple(lerp(v0, v1, u))
    bot = tuple(lerp(v3, v2, u))
    return tuple(lerp(top, bot, v))


def subdivide_quad(v0, v1, v2, v3, nx, ny, color_a, color_b):
    """Subdivide a 3D quad into nx*ny smaller quads with checkerboard colours.

    Returns list of ((cx,cy,cz),  (p00,p10,p11,p01,  colour)).
    The centre-of-mass is used for depth sorting.
    """
    patches = []
    for j in range(ny):
        for i in range(nx):
            u0, u1 = i / nx, (i + 1) / nx
            v0_, v1_ = j / ny, (j + 1) / ny

            p00 = bilerp(v0, v1, v2, v3, u0, v0_)
            p10 = bilerp(v0, v1, v2, v3, u1, v0_)
            p11 = bilerp(v0, v1, v2, v3, u1, v1_)
            p01 = bilerp(v0, v1, v2, v3, u0, v1_)

            cx = (p00[0] + p10[0] + p11[0] + p01[0]) / 4
            cy = (p00[1] + p10[1] + p11[1] + p01[1]) / 4
            cz = (p00[2] + p10[2] + p11[2] + p01[2]) / 4

            colour = color_a if (i + j) % 2 == 0 else color_b
            patches.append(((cx, cy, cz), (p00, p10, p11, p01, colour)))
    return patches


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_view(rot_y_deg, rot_x_deg, label=""):
    """Render the box with given rotations (in degrees)."""
    ry = math.radians(rot_y_deg)
    rx = math.radians(rot_x_deg)

    # Rotate each cube vertex
    rotated = [rot_x(rot_y(p, ry), rx) for p in CUBE]

    # Build checkerboard patches for every face
    patches = []  # each: ((cx,cy,cz), (p3d..., (r,g,b)))
    for idxs, ca, cb, fname in FACES:
        v0, v1, v2, v3 = [rotated[i] for i in idxs]
        face_patches = subdivide_quad(v0, v1, v2, v3, CHECKER_N, CHECKER_N, ca, cb)
        patches.extend(face_patches)

    # Depth sort: farthest first (painter's algorithm)
    patches.sort(key=lambda x: x[0][2], reverse=True)

    # Create image
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Project and draw each patch
    for _, (p00, p10, p11, p01, colour) in patches:
        poly = [
            project(p00),
            project(p10),
            project(p11),
            project(p01),
        ]
        draw.polygon(poly, fill=colour)

    # Draw face labels at the centre of each face
    # (only for faces pointing roughly toward camera — z<0 means facing us)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idxs, ca, cb, fname in FACES:
        v0, v1, v2, v3 = [rotated[i] for i in idxs]
        cx = (v0[0] + v1[0] + v2[0] + v3[0]) / 4
        cy_ = (v0[1] + v1[1] + v2[1] + v3[1]) / 4
        cz = (v0[2] + v1[2] + v2[2] + v3[2]) / 4
        # Draw label only if face is roughly front-facing
        if cz < 0:
            px, py = project((cx, cy_, cz))
            # Light outline for readability
            draw.text((px - 1, py - 1), fname, fill=(0, 0, 0), font=font, anchor="mm")
            draw.text((px + 1, py + 1), fname, fill=(0, 0, 0), font=font, anchor="mm")
            draw.text((px, py), fname, fill=(255, 255, 255), font=font, anchor="mm")

    if label:
        draw.text((10, 10), label, fill=(80, 80, 80), font=font)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    out_dir = os.path.dirname(os.path.abspath(__file__))

    img1 = render_view(rot_y_deg=-25, rot_x_deg=-15, label="View 1")
    img1.save(os.path.join(out_dir, "box_00.png"), "PNG")
    print(f"Saved {out_dir}/box_00.png  ({img1.size[0]}x{img1.size[1]})")

    img2 = render_view(rot_y_deg=-33, rot_x_deg=-20, label="View 2")
    img2.save(os.path.join(out_dir, "box_01.png"), "PNG")
    print(f"Saved {out_dir}/box_01.png  ({img2.size[0]}x{img2.size[1]})")

    print("Done.")
