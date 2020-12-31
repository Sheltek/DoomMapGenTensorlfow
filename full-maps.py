#!/usr/bin/python
# Original from Fredrik Johansson, 2006-12-11 
# Incorporates code by Frans P. de Vries, 2016-04-26 
# Incorporates code by Wintermute0110, 2017-03-11
#
# Image size table:
#   1080p -- 1920 x 1080
#    720p -- 1280 x  720
#    576p --  720 x  576
#    480p --  720 x  480
#
# Ideas for map drawing. This allows to have several maps per level for extrafanart.
#   a) LINEDEFS: plot linedefs and change colour for two sided/action linedefs
#   b) SECTORS:  plot sectors (sectors with same height have same colour)
#   c) NODES:    plot ssegs and ssectors
#   d) VERTEXES: plot linedef vertices in one color and node-builder-added vertices in another
#
# For the future:
#   a) TEXTURED: plot sector floor textures (same as automap in GLPrBoom+)
#
import sys
import os
import getopt
import pprint
import math
from PIL import Image, ImageDraw

# --- Add OMG module to Python path ---

real_path = os.path.realpath(__file__)
currentdir = os.path.dirname(real_path)
parentdir = os.path.dirname(currentdir)
moduledir = os.path.dirname(parentdir)
sys.path.insert(0, moduledir)
from omg import *

# -------------------------------------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------------------------------------
VERBOSE = True
BORDER_PERCENT = 8

MAP_LINEDEFS = 100
MAP_SECTORS = 200
MAP_NODES = 300
MAP_VERTEXES = 400

def cmp(a, b):
    return (a > b) - (a < b)

# -------------------------------------------------------------------------------------------------
# Vars
# -------------------------------------------------------------------------------------------------
class LinearTransform:
    def __init__(self, left, right, bottom, top, px_size, py_size, border):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.x_size = right - left
        self.y_size = top - bottom
        # --- Shift map in x or y direction ---
        self.pan_x = 0
        self.pan_y = 0

        print('LinearTransform() left       = {0}'.format(left))
        print('LinearTransform() right      = {0}'.format(right))
        print('LinearTransform() bottom     = {0}'.format(bottom))
        print('LinearTransform() top        = {0}'.format(top))
        print('LinearTransform() x_size     = {0}'.format(self.x_size))
        print('LinearTransform() y_size     = {0}'.format(self.y_size))

        # --- Calculate scale in [pixels] / [map_unit] ---
        self.px_size = px_size
        self.py_size = py_size
        self.border = border
        self.border_x = px_size * border / 100
        self.border_y = py_size * border / 100
        self.pxsize_nob = px_size - 2 * self.border_x
        self.pysize_nob = py_size - 2 * self.border_y
        self.x_scale = self.pxsize_nob / float(self.x_size)
        self.y_scale = self.pysize_nob / float(self.y_size)
        if self.x_scale < self.y_scale:
            self.scale = self.x_scale
            self.xoffset = self.border_x
            self.yoffset = (py_size - int(self.y_size * self.scale)) / 2
        else:
            self.scale = self.y_scale
            self.xoffset = (px_size - int(self.x_size * self.scale)) / 2
            self.yoffset = self.border_y
        print('LinearTransform() px_size    = {0}'.format(px_size))
        print('LinearTransform() py_size    = {0}'.format(py_size))
        print('LinearTransform() border     = {0}'.format(border))
        print('LinearTransform() border_x   = {0}'.format(self.border_x))
        print('LinearTransform() border_y   = {0}'.format(self.border_y))
        print('LinearTransform() pxsize_nob = {0}'.format(self.pxsize_nob))
        print('LinearTransform() pysize_nob = {0}'.format(self.pysize_nob))
        print('LinearTransform() xscale     = {0}'.format(self.x_scale))
        print('LinearTransform() yscale     = {0}'.format(self.y_scale))
        print('LinearTransform() scale      = {0}'.format(self.scale))
        print('LinearTransform() xoffset    = {0}'.format(self.xoffset))
        print('LinearTransform() yoffset    = {0}'.format(self.yoffset))

    def MapToScreen(self, map_x, map_y):
        screen_x = self.scale * (+map_x - self.left) + self.xoffset
        screen_y = self.scale * (-map_y + self.top) + self.yoffset

        return (int(screen_x), int(screen_y))

    def ScreenToMap(self, screen_x, screen_y):
        map_x = +(screen_x - self.xoffset + self.scale * self.left) / self.scale
        map_y = -(screen_y - self.yoffset - self.scale * self.top) / self.scale

        return (int(map_x), int(map_y))


# See https://github.com/chocolate-doom/chocolate-doom/blob/sdl2-branch/src/doom/am_map.c
class ColorScheme:
    def __init__(self, back, wall, tswall, awall, fdwall, cdwall, thing):
        self.BG = back
        self.WALL = wall  # One sided linedef
        self.TS_WALL = tswall  # Two sided linedef
        self.A_WALL = awall  # Action wall
        self.FD_WALL = fdwall  # Two sided, floor level change
        self.CD_WALL = cdwall  # Two sided, ceiling level change and same floor level
        self.THING = thing  # Thing color


CDoomWorld = ColorScheme(
    (255, 255, 255),
    (0, 0, 0),
    (144, 144, 144),
    (0, 255, 0),
    (0, 0, 255),
    (220, 130, 50),
    (0, 255, 0)
)

CClassic = ColorScheme(
    (0, 0, 0),  # BACK black
    (255, 0, 0),  # WALL red
    (150, 150, 150),  # TS_WALL grey
    (255, 255, 255),  # A_WALL white
    (139, 92, 55),  # FD_WALL brown
    (255, 255, 0),  # CD_WALL yellow
    (220, 130, 50),  # THING green
)

sector_colours = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]
num_sector_colours = len(sector_colours)


# -------------------------------------------------------------------------------------------------
# Drawing utility functions
# -------------------------------------------------------------------------------------------------
def draw_line(draw, p1x, p1y, p2x, p2y, color):
    draw.line((p1x, p1y, p2x, p2y), fill=color)


def draw_thick_line(draw, p1x, p1y, p2x, p2y, color):
    draw.line((p1x, p1y, p2x, p2y), fill=color)
    draw.line((p1x + 1, p1y, p2x + 1, p2y), fill=color)
    draw.line((p1x - 1, p1y, p2x - 1, p2y), fill=color)
    draw.line((p1x, p1y + 1, p2x, p2y + 1), fill=color)
    draw.line((p1x, p1y - 1, p2x, p2y - 1), fill=color)


def draw_axis(draw, LT, color):
    (pxzero, pyzero) = LT.MapToScreen(0, 0)
    draw.line((0, pyzero, LT.px_size, pyzero), fill=color)
    draw.line((pxzero, 0, pxzero, LT.py_size), fill=color)


#
# Draw the grid every 128 map units with origin at (0, 0)
# I think the blockmap is similar to this grid.
# Vanilla comment: "Draws flat (floor/ceiling tile) aligned grid lines."
# See https://github.com/chocolate-doom/chocolate-doom/blob/sdl2-branch/src/doom/am_map.c#L1100
#
def draw_grid(draw, LT, color):
    # --- Draw vertical gridlines ---
    start = LT.left
    # if (start - bmaporgx) % 128:
    #     start += 128 - ((start-bmaporgx) % 128)
    end = LT.right
    for x in range(start, end, 128):
        (A_px, A_py) = LT.MapToScreen(x, LT.bottom)
        (B_px, B_py) = LT.MapToScreen(x, LT.top)
        draw.line((A_px, A_py, B_px, B_py), fill=color)

    # --- Draw horizontal gridlines ---
    start = LT.bottom
    end = LT.top
    for x in range(start, end, 128):
        (A_px, A_py) = LT.MapToScreen(LT.left, x)
        (B_px, B_py) = LT.MapToScreen(LT.right, x)
        draw.line((A_px, A_py, B_px, B_py), fill=color)


#
# A level must be contained within a 16384-unit radius as measured from its center point.
# Point A is the top-left corner.
#
# A---------B---------C   A-C gap 327 map units
# |         |         |   A-B gap 163 map units
# |         E         |   A-D gap is 163/2 map units
# D                   F   B-E gap is 163/4 map units
#
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
    draw.line((C_px, C_py, F_px, F_py), fill=color)  # C -> F


#
# Draw a triangle with same size as in Vanilla Doom
# https://github.com/chocolate-doom/chocolate-doom/blob/sdl2-branch/src/doom/am_map.c#L186
# https://github.com/chocolate-doom/chocolate-doom/blob/sdl2-branch/src/doom/am_map.c#L1314
#
thintriangle_guy = [
    [[-8, -11.2], [16, 0.0]],
    [[16, 0.0], [-8, 11.2]],
    [[-8, 11.2], [-8, -11.2]]
]


def draw_thing(draw, LT, map_x, map_y, angle, color):
    angle_rad = math.radians(angle)
    for line in thintriangle_guy:
        # -- Rotate ---
        rot_a_x = line[0][0] * math.cos(angle_rad) - line[0][1] * math.sin(angle_rad)
        rot_a_y = line[0][0] * math.sin(angle_rad) + line[0][1] * math.cos(angle_rad)
        rot_b_x = line[1][0] * math.cos(angle_rad) - line[1][1] * math.sin(angle_rad)
        rot_b_y = line[1][0] * math.sin(angle_rad) + line[1][1] * math.cos(angle_rad)

        # --- Translate to thing coordinates on map ---
        A_x = rot_a_x + map_x
        A_y = rot_a_y + map_y
        B_x = rot_b_x + map_x
        B_y = rot_b_y + map_y

        # --- Draw line ---
        (A_px, A_py) = LT.MapToScreen(A_x, A_y)
        (B_px, B_py) = LT.MapToScreen(B_x, B_y)
        draw.line((A_px, A_py, B_px, B_py), fill=(0, 255, 0))


# -------------------------------------------------------------------------------------------------
# Top level map drawing functions
# -------------------------------------------------------------------------------------------------
def drawmap_fit(wad, map_name, filename, format, map_type, px_size, py_size, cscheme):
    # --- Load map in editor ---
    edit = MapEditor(wad.maps[map_name])

    # --- Determine scale = pixel / map unit ---
    xmin = min([v.x for v in edit.vertexes])
    xmax = max([v.x for v in edit.vertexes])
    ymin = min([v.y for v in edit.vertexes])
    ymax = max([v.y for v in edit.vertexes])
    LT = LinearTransform(xmin, xmax, ymin, ymax, px_size, py_size, BORDER_PERCENT)

    # --- Create image ---
    im = Image.new('RGB', (px_size, py_size), cscheme.BG)
    draw = ImageDraw.Draw(im)

    # --- Draw XY axis ---
    # draw_grid(draw, LT, (100, 100, 100))
    # draw_axis(draw, LT, (200, 200, 200))

    # --- Draw map scale ---
    draw_scale(draw, LT, (256, 256, 256))

    # --- Draw map ---
    if map_type == MAP_LINEDEFS:
        # --- sorts lines ---
        edit.linedefs.sort(lambda a, b: cmp(not a.two_sided, not b.two_sided))

        # --- Draw lines ---
        if False:
            # >> Use DoomWiki colours and drawing algorithm
            for line in edit.linedefs:
                (p1x, p1y) = LT.MapToScreen(edit.vertexes[line.vx_a].x, edit.vertexes[line.vx_a].y)
                (p2x, p2y) = LT.MapToScreen(edit.vertexes[line.vx_b].x, edit.vertexes[line.vx_b].y)

                color = cscheme.WALL
                if line.two_sided:
                    color = cscheme.TS_WALL
                elif line.action:
                    color = cscheme.A_WALL

                # >> Draw several lines to simulate thickness
                # draw_line(draw, p1x, p1y, p2x, p2y, color)
                draw_thick_line(draw, p1x, p1y, p2x, p2y, color)
        else:
            # --- Create floorheight and ceilingheight for two-sided linedefs ---
            for line in edit.linedefs:
                # Skin one-sided linedefs
                if line.back < 0: continue
                front_sidedef = edit.sidedefs[line.front]
                back_sidedef = edit.sidedefs[line.back]
                front_sector = edit.sectors[front_sidedef.sector]
                back_sector = edit.sectors[back_sidedef.sector]
                line.frontsector_floorheight = front_sector.z_floor
                line.frontsector_ceilingheight = front_sector.z_ceil
                line.backsector_floorheight = back_sector.z_floor
                line.backsector_ceilingheight = back_sector.z_ceil

            # >> Use Vanilla Doom automap colours and drawing algortihm
            # >> https://github.com/chocolate-doom/chocolate-doom/blob/sdl2-branch/src/doom/am_map.c#L1146
            for line in edit.linedefs:
                (p1x, p1y) = LT.MapToScreen(edit.vertexes[line.vx_a].x, edit.vertexes[line.vx_a].y)
                (p2x, p2y) = LT.MapToScreen(edit.vertexes[line.vx_b].x, edit.vertexes[line.vx_b].y)

                # >> Use same algorithm as AM_drawWalls(). cheating variable is true
                # >> In vanilla secret walls have same colours as walls.
                if line.back < 0:
                    color = cscheme.WALL
                elif line.backsector_floorheight != line.frontsector_floorheight:
                    color = cscheme.FD_WALL
                elif line.backsector_ceilingheight != line.frontsector_ceilingheight:
                    color = cscheme.CD_WALL
                else:
                    color = cscheme.TS_WALL

                # >> Draw several lines to simulate thickness
                # draw_line(draw, p1x, p1y, p2x, p2y, color)
                draw_thick_line(draw, p1x, p1y, p2x, p2y, color)

        # --- Draw things ---
        for thing in edit.things:
            draw_thing(draw, LT, thing.x, thing.y, thing.angle, cscheme.THING)

    elif map_type == MAP_VERTEXES:
        # --- Draw lines ---
        edit.linedefs.sort(lambda a, b: cmp(not a.two_sided, not b.two_sided))
        for line in edit.linedefs:
            (p1x, p1y) = LT.MapToScreen(edit.vertexes[line.vx_a].x, edit.vertexes[line.vx_a].y)
            (p2x, p2y) = LT.MapToScreen(edit.vertexes[line.vx_b].x, edit.vertexes[line.vx_b].y)
            color = cscheme.WALL
            draw_line(draw, p1x, p1y, p2x, p2y, color)

        # --- Plot vertices not belonging to a linedef ---
        SIZE = 2
        linedefs_vertexes = set()
        for line in edit.linedefs:
            linedefs_vertexes.add(line.vx_a)
            linedefs_vertexes.add(line.vx_b)
        for v_index in range(len(edit.vertexes)):
            # print(vars(vertex))
            color = (0, 255, 255)
            if v_index not in linedefs_vertexes: color = (255, 255, 0)
            (p1x, p1y) = LT.MapToScreen(edit.vertexes[v_index].x, edit.vertexes[v_index].y)
            draw.rectangle((p1x - SIZE, p1y - SIZE, p1x + SIZE, p1y + SIZE), fill=color)

    elif map_type == MAP_SECTORS:
        # Approach A: paint each sector surface with a different colour. Colours will be picked
        #             sequentially from a list.
        # Approach B: paint the floor texture of each sector, aligned to the 64x64 grid.

        # --- Make a list of sectors. Each sector has a list of linedef numbers ---
        num_linedefs = len(edit.linedefs)
        num_sidedefs = len(edit.sidedefs)
        num_sectors = len(edit.sectors)
        print('Map has {0} linedefs'.format(num_linedefs))
        print('Map has {0} sidedefs'.format(num_sidedefs))
        print('Map has {0} sectors'.format(num_sectors))
        sector_list = [list() for _ in range(num_sectors)]
        for i in range(num_linedefs):
            line = edit.linedefs[i]
            # print(vars(line))
            front = line.front
            back = line.back
            front_sector = edit.sidedefs[front].sector
            sector_list[front_sector].append(i)
            if back > 0:
                back_sector = edit.sidedefs[back].sector
                sector_list[back_sector].append(i)
        for i in range(num_sectors):
            print('Sector {0:4d} -> {1}'.format(i, sector_list[i]))

        # --- Draw sectors ---
        for i in range(num_sectors):
            # for i in [4]:
            sector = sector_list[i]
            print('Processing sector {0} ...'.format(i))

            # --- Get sector bouding box ---
            s_left = s_bottom = 100000
            s_right = s_top = -100000
            sector_cord_list = []
            for linedef_num in sector:
                line = edit.linedefs[linedef_num]
                x_cord = edit.vertexes[line.vx_a].x
                y_cord = edit.vertexes[line.vx_a].y
                if x_cord < s_left:   s_left = x_cord
                if x_cord > s_right:  s_right = x_cord
                if y_cord < s_bottom: s_bottom = y_cord
                if y_cord > s_top:    s_top = y_cord
                sector_cord_list.append((x_cord, y_cord))
            s_xsize = s_right - s_left
            s_ysize = s_top - s_bottom
            print('left {0} | right {1} | bottom {2} | top {3}'.format(s_left, s_right, s_bottom, s_top))
            print('xsize {0} | ysize {1}'.format(s_xsize, s_ysize))

            # --- Transform sector coordinates to unscaled pixels ---
            s_pos_vector = LT.MapToScreen(s_left, s_top)
            s_screen_xsize = s_xsize * LT.scale
            s_screen_ysize = s_ysize * LT.scale
            s_LT = LinearTransform(s_left, s_right, s_bottom, s_top, s_screen_xsize, s_screen_ysize, 0)
            sector_pixel_cord_list = []
            for t in sector_cord_list:
                sector_pixel_cord_list.append(s_LT.MapToScreen(t[0], t[1]))
            print(s_pos_vector)
            print(sector_cord_list)
            print(sector_pixel_cord_list)

            # --- Create a sector square image ---
            # http://stackoverflow.com/questions/3119999/drawing-semi-transparent-polygons-in-pil
            s_poly = Image.new('RGB', (s_xsize, s_ysize))
            poly_draw = ImageDraw.Draw(s_poly)
            colour_index = i % num_sector_colours
            poly_draw.polygon(sector_pixel_cord_list, fill=sector_colours[colour_index],
                              outline=sector_colours[colour_index])
            del poly_draw
            s_poly.save('temp.png', 'PNG')

            # s_back = Image.new('RGB', (s_xsize, s_ysize), (255, 0, 0))
            # s_back.paste(s_poly, mask = s_poly)
            im.paste(s_poly, box=s_pos_vector)


    elif map_type == MAP_NODES:
        # NOTE The implicit lines of the subsectors are the partition lines of the nodes.
        #
        # --- Draw segs and ssectors ---
        # NOT WORKING AT THE MOMENT!
        # Segs are only defined on linedefs and have implicit edges. I don't know how to
        # plot the implicit edges.
        # http://doom.wikia.com/wiki/Subsector
        # http://doom.wikia.com/wiki/User_talk:Fraggle#Making_polygons
        # https://www.doomworld.com/vb/doom-editing/43976-subsector-viewer/
        # https://www.doomworld.com/vb/doom-general/70565-quesiton-about-the-nodes-lump-and-subsectors/
        #
        print('{0} segs'.format(len(edit.segs)))
        print('{0} linedefs'.format(len(edit.linedefs)))

        s_left = s_bottom = 100000
        s_right = s_top = -100000

        # --- Draw linedefs ---
        for line in edit.linedefs:
            p1x = LT.scale * (edit.vertexes[line.vx_a].x - s_left) + LT.xoffset
            p1y = LT.scale * (edit.vertexes[line.vx_a].y - s_bottom) + LT.yoffset
            p2x = LT.scale * (edit.vertexes[line.vx_b].x - s_left) + LT.xoffset
            p2y = LT.scale * (edit.vertexes[line.vx_b].y - s_bottom) + LT.yoffset
            color = cscheme.WALL
            draw_line(draw, p1x, p1y, p2x, p2y, color)

        # --- Draw segs ---
        for seg in edit.segs:
            line = edit.linedefs[seg.line]
            # print(vars(seg))
            # print(vars(line))

            seg_vx_a = seg.vx_a
            seg_vx_b = seg.vx_b
            line_vx_a = line.vx_a
            line_vx_b = line.vx_b

            if seg_vx_a != line_vx_a or seg_vx_b != line_vx_b:
                print('seg_vx_a  {0:5d}   seg_vx_b {1:5d}'.format(seg_vx_a, seg_vx_b))
                print('line_vx_a {0:5d}  line_vx_b {1:5d}'.format(line_vx_a, line_vx_b))

                p1x = LT.scale * (edit.vertexes[seg.vx_a].x - s_left) + LT.xoffset
                p1y = LT.scale * (edit.vertexes[seg.vx_a].y - s_bottom) + LT.yoffset
                p2x = LT.scale * (edit.vertexes[seg.vx_b].x - s_left) + LT.xoffset
                p2y = LT.scale * (edit.vertexes[seg.vx_b].y - s_bottom) + LT.yoffset
                color = cscheme.FDWALL
                draw_line(draw, p1x, p1y, p2x, p2y, color)

        # --- Plot vertices not belonging to a linedef ---
        SIZE = 2
        linedefs_vertexes = set()
        for line in edit.linedefs:
            linedefs_vertexes.add(line.vx_a)
            linedefs_vertexes.add(line.vx_b)
        for v_index in range(len(edit.vertexes)):
            # print(vars(vertex))
            color = (0, 255, 255)
            if v_index not in linedefs_vertexes: color = (255, 255, 0)
            p1x = LT.scale * (edit.vertexes[v_index].x - s_left) + LT.xoffset
            p1y = LT.scale * (edit.vertexes[v_index].y - s_bottom) + LT.yoffset
            draw.point((p1x, p1y), fill=color)
            draw.rectangle((p1x - SIZE, p1y - SIZE, p1x + SIZE, p1y + SIZE), fill=color)

    # --- Save image file ---
    del draw
    im.save(filename, format)


# -------------------------------------------------------------------------------------------------
# main ()
# -------------------------------------------------------------------------------------------------
if len(sys.argv) < 2:
    print('Omgifol DEMO script: draw maps to image files')
    print('Draw all maps whose names match the given pattern (eg E?M4 or MAP*)')
    print('to image files of a given format (PNG, BMP, etc). width specifies the')
    print('desired width of the output images.\n')
    print('Usage: drawmaps.py [options] source.wad\n')
    print('  -p pattern  Patterns may be "E?M?", "MAP01", "MAP*". Defaults to all level (pattern "*")')
    print('  -f format   May be PNG, BMP, JPEG. Defaults to PNG')
    print('  -w width    Width in pixels. Defaults to 1920 (for a 1920x1080 image)')
    print('  -s size     Fits map on given image size. size may be 1080p, 720p, 576p, 480p')
    print('  -things     Draws things on maps')
    print('  -axis       Draws cartesian axis on maps')
    print('  -scale      Draws a map scale on the upper right corner')

    sys.exit(1)

# --- Parse arguments ---
opts, args = getopt.getopt(sys.argv[1:], 'p:w:f:')
pprint.pprint(opts)
pprint.pprint(args)
wad_filename = args[0]
pattern = '*'
width = 1920
format = 'PNG'
for o, a in opts:
    if o == '-p':
        pattern = a
    elif o == '-f':
        format = a
    elif o == '-w':
        width = int(a)
    else:
        assert False, "Unhandled option"

# --- Load WAD ---
print('Loading WAD "{0}" ...'.format(wad_filename))
inwad = WAD()
inwad.from_file(wad_filename)
for name in inwad.maps.find(pattern):
    fn_prefix = os.path.splitext(wad_filename)[0] + '_' + name
    fn_sufix = '.' + format.lower()
    print('Drawing map {0} on file "{1}"'.format(name, fn_prefix))
    # drawmap_fit(inwad, name, fn_prefix + '_Line_A' + fn_sufix, format, MAP_LINEDEFS, 1920, 1080, CDoomWorld)
    drawmap_fit(inwad, name, fn_prefix + '_Line_B' + fn_sufix, format, MAP_LINEDEFS, 1920, 1080, CClassic)
    # drawmap_fit(inwad, name, fn_prefix + '_Ver_A' + fn_sufix, format, MAP_VERTEXES, 1920, 1080, CDoomWorld)
    # drawmap_fit(inwad, name, fn_prefix + '_Ver_B' + fn_sufix, format, MAP_VERTEXES, 1920, 1080, CClassic)
    # drawmap_fit(inwad, name, fn_prefix + '_Sec_A' + fn_sufix, format, MAP_SECTORS, 1920, 1080, CDoomWorld)
    drawmap_fit(inwad, name, fn_prefix + '_Sec_B' + fn_sufix, format, MAP_SECTORS, 1920, 1080, CClassic)
    # drawmap_fit(inwad, name, fn_prefix + '_Nodes_A' + fn_sufix, format, MAP_NODES, 1920, 1080, CDoomWorld)
    # drawmap_fit(inwad, name, fn_prefix + '_Nodes_B' + fn_sufix, format, MAP_NODES, 1920, 1080, CClassic)
sys.exit(0)
