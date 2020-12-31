from __future__ import print_function

from LinearTransform import LinearTransform
from omg import *
import sys
from PIL import Image, ImageDraw


def drawmap(wad, width, format):
    xsize = width - 8

    i = 77
    for level in wad.maps:
        edit = MapEditor(wad.maps[level])
        xmin = ymin = 32767
        xmax = ymax = -32768
        for v in edit.vertexes:
            xmin = min(xmin, v.x)
            xmax = max(xmax, v.x)
            ymin = min(ymin, -v.y)
            ymax = max(ymax, -v.y)

        scale = xsize / float(xmax - xmin)
        xmax = int(xmax * scale)
        xmin = int(xmin * scale)
        ymax = int(ymax * scale)
        ymin = int(ymin * scale)

        for v in edit.vertexes:
            v.x = v.x * scale
            v.y = -v.y * scale

        im = Image.new('RGB', ((xmax - xmin) + 8, (ymax - ymin) + 8), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        edit.linedefs.sort(key=lambda a: not a.two_sided)

        for line in edit.linedefs:
            p1x = edit.vertexes[line.vx_a].x - xmin + 8
            p1y = edit.vertexes[line.vx_a].y - ymin + 8
            p2x = edit.vertexes[line.vx_b].x - xmin + 8
            p2y = edit.vertexes[line.vx_b].y - ymin + 8

            color = (255, 255, 255)
            if line.two_sided:
                color = (255, 255, 255)
            if line.action:
                color = (255, 255, 255)

            draw.line((p1x, p1y, p2x, p2y), fill=color)

        del draw

        im.save(str(i) + "." + format.lower(), format)
        i += 1


# import psyco
# psyco.full()

def draw_scale(draw, LT, color):
    (A_px, A_py) = LT.MapToScreen(LT.right - 256, LT.top)
    (B_px, B_py) = LT.MapToScreen(LT.right - 128, LT.top)
    (C_px, C_py) = LT.MapToScreen(LT.right, LT.top)
    (D_px, D_py) = LT.MapToScreen(LT.right - 256, LT.top - 128 / 2)
    (E_px, E_py) = LT.MapToScreen(LT.right - 128, LT.top - 128 / 4)
    (F_px, F_py) = LT.MapToScreen(LT.right, LT.top - 128 / 2)

    draw.line((A_px, A_py, C_px, C_py), fill=color)  # A -> C
    draw.line((A_px, A_py, D_px, D_py), fill=color)  # A -> D
    draw.line((B_px, B_py, E_px, E_py), fill=color)  # B -> E


def draw_line(draw, p1x, p1y, p2x, p2y, color):
    draw.line((p1x, p1y, p2x, p2y), fill=color)


if (len(sys.argv) < 5):
    print("\n    Omgifol script: draw maps to image files\n")
    print("    Usage:")
    print("    drawmaps.py source.wad pattern width format\n")
    print("    Draw all maps whose names match the given pattern (eg E?M4 or MAP*)")
    print("    to image files of a given format (PNG, BMP, etc). width specifies the")
    print("    desired width of the output images.")
else:

    print("Loading %s..." % sys.argv[1])
    inwad = WAD()
    inwad.from_file(sys.argv[1])
    width = int(sys.argv[3])
    format = sys.argv[4].upper()
    drawmap(inwad, width, format)
