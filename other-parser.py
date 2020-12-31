#!/usr/bin/python3

# import png

import itertools
import time
import numpy as np
import math
import random
import os
import sys
import struct
# import pprint
import re
from fractions import Fraction, gcd
import scipy.ndimage as ndimage

# import svgwrite

import OpenGL.GL as gl
import OpenGL.GLU as glu
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import pygame

STEREOSCOPIC = False


def normalize(x):
    n = np.linalg.norm(x)
    if n != 0:
        return x / n
    else:
        return x


def dot_product(a, b):
    return sum(x * y for (x, y) in zip(a, b))


def line_distance(l, a):
    return dot_product(l['normal'], a) - l['dist']


def segment_length_squared(s):
    return sum([(x - y) * (x - y) for (x, y) in zip(s[0], s[1])])


def normalize_integer(x):
    return (1 if x > 0 else 0) - (1 if x < 0 else 0)


def intersect_lines(a, b):
    det = a['normal'][0] * b['normal'][1] - a['normal'][1] * b['normal'][0]
    detx = a['dist'] * b['normal'][1] - a['normal'][1] * b['dist']
    dety = a['normal'][0] * b['dist'] - a['dist'] * b['normal'][0]
    if abs(det) == 1:
        return [detx * det, dety * det]
    elif det != 0:
        return [Fraction(detx, det), Fraction(dety, det)]
    else:
        return None


def make_line(seg):
    t = [b - a for (b, a) in zip(seg[1], seg[0])]
    # The normal points toward the left of the directed segment starting with the endpoint seg[0]
    n = [-t[1], t[0]]
    return {'normal': n, 'tangent': t, 'dist': sum([a * b for (a, b) in zip(n, seg[0])])}


def split_polygon(invertexes, splitter):
    vertexes = [[], []]

    circular_vertexes = invertexes + invertexes[:1]

    previous_side = None
    for v in reversed(invertexes):
        previous_side = normalize_integer(line_distance(splitter, v))
        if previous_side != 0:
            break

    # print('splitter: {}'.format(splitter_edge['name']))
    # print('split: {}, {}'.format(splitter_edge, splitter))
    for i in range(len(invertexes)):
        vertex = circular_vertexes[i:i + 2]
        line = make_line(vertex)
        intersection = intersect_lines(splitter, line)

        # Determine which side of the splitter the current vertex is on.
        sides = [normalize_integer(line_distance(splitter, v)) for v in vertex]
        if sides[0] != 0:
            side_index = (sides[0] + 1) // 2
        else:
            side_index = (previous_side + 1) // 2

        # If there was an intersection on the line segment, then add it to both output polygons.
        if intersection is not None and sides[0] * sides[1] < 0:
            # if sides[0]*sides[1] < 0:
            # Add the vertex to the output polygon that is on the same side of the splitter. Annotate the edge with the current splitter.
            vertexes[side_index].append(vertex[0])

            # Add the intersection to the polygon on the other side of the splitter. The edge is the originating edge and is also annotated with the splitter.
            vertexes[1 - side_index].append(intersection)

            # Add the intersection to the polygon on the same side of the splitter. The edge is the splitter edge.
            vertexes[side_index].append(intersection)

            previous_side = sides[0]
        elif sides[0] * sides[1] > 0 or sides[0] != 0:
            # Add the vertex to the output polygon that is on the same side of the splitter.
            vertexes[side_index].append(vertex[0])
            previous_side = sides[0]
        elif sides[0] == 0 and sides[1] == 0:
            vertexes[1 - side_index].append(vertex[0])

            # The vertex is on the splitter. Add it to the polygon on the same side of the splitter and annotate its edge with the splitter.
            vertexes[side_index].append(vertex[0])
        elif sides[0] == 0 and sides[1] != 0:
            if previous_side == sides[1]:
                vertexes[side_index].append(vertex[0])
            else:
                vertexes[side_index].append(vertex[0])

                vertexes[1 - side_index].append(vertex[0])

    return tuple(vertexes)


def vector_add(p0, p1):
    return tuple([a + b for (a, b) in zip(list(p0), list(p1))])


def rotation_matrix(axis, theta):
    a = math.cos(theta / 2)
    b, c, d = -normalize(axis) * math.sin(theta / 2)
    return np.array(
        [[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
         [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
         [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


ANGLE_TO_RADIANS = math.pi / 180


class Player:
    UP_VECTOR = np.array([0, 0, 1])

    def __init__(self, viewpoint=[0.0, 0.0, 0.0], direction=0.0, pitch=0.0):
        cs = math.cos(direction)
        sn = math.sin(direction)
        pcs = math.cos(pitch)
        psn = math.sin(pitch)
        self.direction = normalize(np.array([cs * pcs, sn * pcs, psn], dtype=np.float64))
        self.viewpoint = np.array(viewpoint, dtype=np.float64)

    def move_forward(self, step):
        self.viewpoint += self.direction * step

    def move_left(self, step):
        left_vector = np.cross(self.direction, -self.UP_VECTOR)
        self.viewpoint += left_vector * step

    def move_up(self, step):
        self.viewpoint += self.UP_VECTOR * step

    def turn_left(self, angle):
        self.direction = np.dot(rotation_matrix(self.UP_VECTOR, -angle), self.direction)

    def look_up(self, angle):
        right_vector = np.cross(self.direction, self.UP_VECTOR)
        new_direction = np.dot(rotation_matrix(right_vector, -angle), self.direction)
        if abs(np.dot(new_direction, self.UP_VECTOR)) < (1 - 1e-2):
            self.direction = new_direction

    def update_OpenGL(self, iod=0, focal_distance=None):
        scale_factor = 1 / 100.0
        right_vector = np.cross(self.direction, self.UP_VECTOR)
        if focal_distance is not None:
            glu.gluLookAt(*tuple(list(self.viewpoint + right_vector * (iod / 2)) +
                                 list(self.direction * focal_distance + self.viewpoint) + list(self.UP_VECTOR)))
        else:
            glu.gluLookAt(*tuple(list(self.viewpoint + right_vector * (iod / 2)) +
                                 list(self.direction + self.viewpoint + right_vector * (iod / 2)) + list(
                self.UP_VECTOR)))

    def pitch_angle(self):
        cs = np.linalg.norm(self.direction[0:1])
        sn = self.direction[2]
        return math.atan2(sn, cs)

    def direction_angle(self):
        cs = self.direction[0]
        sn = self.direction[1]
        angle = math.atan2(sn, cs)
        return angle

    def set_angle(self, direction, pitch=0):
        cs = math.cos(direction)
        sn = math.sin(direction)
        pcs = math.cos(pitch)
        psn = math.sin(pitch)
        self.direction = normalize(np.array([cs * pcs, sn * pcs, psn], dtype=np.float64))

    def set_elevation(self, z):
        self.viewpoint[2] = z

    def get_position(self):
        return self.viewpoint[:]

    def set_position(self, pos):
        self.viewpoint = pos[:]


# The wad file lump strings have a constant length are padded with null bytes at the end.
# This function will convert a lump string into a python string.
def parse_string8(b):
    return b.decode('ascii', 'ignore').partition('\0')[0]


# Load all of the levels in a wad file.
def level_lumps(wad, directory, mapname=None, wadtype='doom'):
    levels = {}
    level = None
    level_name_regex = re.compile('MAP\d\d|E\d\M\d')

    # parse_node_child_link = lambda c: ('subsector', c&0x7fff) if c & 0x8000 != 0 else ('node', c)
    parse_node_child_link = lambda c: ('subsector', c & 0x7fff) if c & 0x8000 != 0 else ('node', c & 0x7fff)
    if wadtype == 'hexen':
        lump_fields = {
            'THINGS': ['tid', 'x_position', 'y_position', 'height', 'angle', 'type', 'options', 'action', 'arguments'],
            'LINEDEFS': ['vertex_start', 'vertex_end', 'flags', 'function', 'arguments', 'sidedef_right',
                         ('sidedef_left', lambda l: None if l == -1 else l)],
            'SIDEDEFS': ['xoffset', 'yoffset', ('uppertexture', parse_string8), ('lowertexture', parse_string8),
                         ('middletexture', parse_string8), 'sector_ref'],
            'VERTEXES': ['X_coord', 'Y_coord'],
            'SEGS': ['vertex_start', 'vertex_end', 'bams', 'line_num', 'segside', 'segoffset'],
            'SSECTORS': ['numsegs', 'start_seg'],
            'NODES': ['x', 'y', 'dx', 'dy', 'box0top', 'box0bottom', 'box0left', 'box0right', 'box1top', 'box1bottom',
                      'box1left', 'box1right', ('child0', parse_node_child_link), ('child1', parse_node_child_link)],
            'SECTORS': ['floorheight', 'ceilingheight', ('floorpic', parse_string8), ('ceilingpic', parse_string8),
                        'lightlevel', 'special_sector', 'tag']
        }
        parse_lump_format = {
            'THINGS': struct.Struct('<7hc5s'),
            'LINEDEFS': struct.Struct('<3hc5s2h'),
            'SIDEDEFS': struct.Struct('<2h8s8s8sh'),
            'VERTEXES': struct.Struct('<2h'),
            'SEGS': struct.Struct('<6h'),
            'SSECTORS': struct.Struct('<2h'),
            'NODES': struct.Struct('<4h8h2H'),
            'SECTORS': struct.Struct('<2h8s8s3h')
        }
    else:
        lump_fields = {
            'THINGS': ['x_position', 'y_position', 'angle', 'type', 'options'],
            'LINEDEFS': ['vertex_start', 'vertex_end', 'flags', 'function', 'tag', 'sidedef_right',
                         ('sidedef_left', lambda l: None if l == -1 else l)],
            'SIDEDEFS': ['xoffset', 'yoffset', ('uppertexture', parse_string8), ('lowertexture', parse_string8),
                         ('middletexture', parse_string8), 'sector_ref'],
            'VERTEXES': ['X_coord', 'Y_coord'],
            'SEGS': ['vertex_start', 'vertex_end', 'bams', 'line_num', 'segside', 'segoffset'],
            'SSECTORS': ['numsegs', 'start_seg'],
            'NODES': ['x', 'y', 'dx', 'dy', 'box0top', 'box0bottom', 'box0left', 'box0right', 'box1top', 'box1bottom',
                      'box1left', 'box1right', ('child0', parse_node_child_link), ('child1', parse_node_child_link)],
            'SECTORS': ['floorheight', 'ceilingheight', ('floorpic', parse_string8), ('ceilingpic', parse_string8),
                        'lightlevel', 'special_sector', 'tag']
        }

        parse_lump_format = {
            'THINGS': struct.Struct('<5h'),
            'LINEDEFS': struct.Struct('<7h'),
            'SIDEDEFS': struct.Struct('<2h8s8s8sh'),
            'VERTEXES': struct.Struct('<2h'),
            'SEGS': struct.Struct('<6h'),
            'SSECTORS': struct.Struct('<2h'),
            'NODES': struct.Struct('<4h8h2H'),
            'SECTORS': struct.Struct('<2h8s8s3h')
        }
    LEVEL_LUMP_NAMES = ['THINGS', 'LINEDEFS', 'SIDEDEFS', 'VERTEXES', 'SEGS', 'SSECTORS', 'NODES', 'SECTORS', 'REJECT',
                        'BLOCKMAP']
    for lump in directory:
        if level_name_regex.match(lump['name']):
            if level and not (mapname is not None and level['name'] != mapname):
                levels[level['name']] = level
            level = {'name': lump['name']}
        # Skip this level if it is not the selected map.
        if mapname is not None and level is not None and level['name'] != mapname:
            continue

        if lump['name'] in LEVEL_LUMP_NAMES and lump['name'] in parse_lump_format:
            lump_name = lump['name']
            parser = parse_lump_format[lump_name]
            fields = lump_fields[lump_name]

            wad.seek(lump['filepos'])
            entries = []
            for i in range(lump['size'] // parser.size):
                values = parser.unpack(wad.read(parser.size))
                entry = {'id': i}
                for (field, value) in zip(fields, values):
                    if isinstance(field, tuple):
                        (fieldname, field_parser) = field
                        entry[fieldname] = field_parser(value)
                    else:
                        entry[field] = value
                entries.append(entry)
            level[lump_name] = entries

            # level[lump['name']] = load_level_lump[lump['name']](lump)
    if level and not (mapname is not None and level['name'] != mapname):
        levels[level['name']] = level
    return levels


def load_colormaps(wad, directory):
    lump = directory['COLORMAP']
    wad.seek(lump['filepos'])
    data = wad.read(256 * 34)

    palette = load_palette(wad, directory)

    colormaps = np.zeros([256, 34, 4], dtype=np.uint8)
    for i in range(34):
        colormap = data[i * 256:(i + 1) * 256]
        # colormaps += [list(colormap)]
        # colormaps[:, i] = list(data[i*256:(i+1)*256])
        for j in range(256):
            colormaps[j, i, :] = palette[colormap[j]]
            # colormaps[j, i, :] = palette[j]

    return colormaps


def load_palette(wad, directory, index=0):
    lump = directory['PLAYPAL']
    wad.seek(lump['filepos'] + index * 256 * 3)
    data = wad.read(256 * 3)
    colors = np.zeros([256, 4], dtype=np.uint8)
    for i in range(256):
        colors[i, :3] = list(data[i * 3:(i + 1) * 3])
        colors[i, 3] = 255
    return colors


def load_flat(wad, lump):
    wad.seek(lump['filepos'])
    data = list(wad.read(64 ** 2))
    return {'width': 64,
            'height': 64,
            'left': 0,
            'top': 0,
            'data': data}


def least_twoexp(x):
    i = 1
    while i < x:
        i *= 2
    return i


def read_texture(wad, directory, texture):
    data = [None] * texture['width'] * texture['height']
    for patch in texture['patches']:
        patch_name = patch['name']
        load_picture(data, wad, directory[patch_name.upper()], (texture['width'], texture['height']), patch['origin'])
    # for i in range(len(data)):
    #    data[i] = random.randint(0, 255)
    return {'width': texture['width'],
            'height': texture['height'],
            'data': data}


def write_texture(tex, name):
    fn = 'textures/{}.png'.format(name.lower())
    wrt = png.Writer(width=tex['width'], height=tex['height'], palette=palette)
    with open(fn, 'wb') as of:
        wrt.write_array(of, [(0 if x is None else x) for x in list(tex['data'])])


def load_picture(data, wad, lump, texsize, origin=(0, 0)):
    wad.seek(lump['filepos'])
    width, height, left, top = struct.unpack('<4h', wad.read(8))
    columns = list(struct.unpack('<{}I'.format(width), wad.read(4 * width)))
    originx, originy = origin
    for i, column in enumerate(columns):
        x = i + originx
        if x < 0: continue
        if x >= texsize[0]: break

        wad.seek(lump['filepos'] + column)
        while True:
            rowstart = wad.read(1)[0]
            if rowstart == 255:
                break
            pixel_count = wad.read(1)[0]
            pixels = wad.read(pixel_count + 2)
            for j in range(pixel_count):
                # x = i+originx+left
                # y = j+rowstart+originy+top
                y = j + rowstart + originy

                if y < 0: continue
                if y >= texsize[1]: break
                data[y * texsize[0] + x] = pixels[j + 1]

    return {'width': width,
            'height': height,
            'left': left,
            'top': top}


def render_level(filename, mapname, wadtype='doom'):
    with open(filename, "rb") as wad:
        magic, numlumps, infotableofs = struct.unpack('<4sII', wad.read(12))
        # print("{} {} {}".format(magic, numlumps, infotableofs))

        # if : print('Hexen!')

        # Load the directory table.
        directory = []
        wad.seek(infotableofs)
        directory_map = {}
        for _ in range(numlumps):
            filepos, size, name = struct.unpack('<II8s', wad.read(16))
            if 'VERBOSE_OUTPUT' in os.environ:
                print("{} at {} with {} bytes".format(parse_string8(name), filepos, size))
            entry = {'filepos': filepos, 'size': size, 'name': parse_string8(name)}
            directory.append(entry)
            directory_map[entry['name']] = entry
        # pprint.pprint(directory)

        # pprint.pprint(directory[start_index:end_index])
        # pprint.pprint(level_lumps(wad, directory, mapname))
        # pprint.pprint(level_lumps(wad, directory, mapname))
        for level in level_lumps(wad, directory, mapname, wadtype).values():
            vertexes = level['VERTEXES']
            segs = level['SEGS']
            linedefs = level['LINEDEFS']
            sidedefs = level['SIDEDEFS']
            nodes = level['NODES']
            subsectors = level['SSECTORS']
            sectors = level['SECTORS']
            things = level['THINGS']
            # pprint.pprint(level)
            for subsector in subsectors:
                s = subsector['start_seg']
                n = subsector['numsegs']
                # pprint.pprint([(seg['vertex_start'], seg['vertex_end']) for seg in segs[s:s+n]])
                # print('\n')
                sector_refs = []
                for seg in segs[s:s + n]:
                    linedef = linedefs[seg['line_num']]

                    sidedef_index = linedef['sidedef_' + ('right' if seg['segside'] == 0 else 'left')]
                    sector_ref = sidedefs[sidedef_index]['sector_ref']
                    sector_refs += [sector_ref]
                subsector['sector_ref'] = sector_ref
                if min(sector_refs) != max(sector_refs):
                    print(sector_refs)
                    sys.exit(0)
            for sector in sectors:
                sector['color'] = tuple([random.randint(0, 100) for _ in range(3)] + ['%'])

            # 'NODES': ['x', 'y', 'dx', 'dy', 'box0top', 'box0bottom', 'box0left', 'box0right', 'box1top', 'box1bottom', 'box1left', 'box1right', ('child0', parse_node_child_link), ('child1', parse_node_child_link)], 

            bbpadding = 16
            bb = [[min([v['X_coord'] for v in vertexes]) - bbpadding,
                   min([v['Y_coord'] for v in vertexes]) - bbpadding],
                  [max([v['X_coord'] for v in vertexes]) + bbpadding,
                   max([v['Y_coord'] for v in vertexes]) + bbpadding]]

            outer_vertexes = list([bb[1], [bb[1][0], bb[0][1]], bb[0], [bb[0][0], bb[1][1]]])
            nodes[-1]['hull'] = outer_vertexes

            # def traverse_node(node):
            for node in reversed(nodes):
                def get_vertex(n, p=''):
                    return tuple([n[p + k] for k in ['x', 'y']])

                node['splitter'] = make_line([
                    get_vertex(node),
                    vector_add(get_vertex(node), get_vertex(node, 'd'))])
                if 'hull' not in node:
                    continue

                for i, sub in enumerate(split_polygon(node['hull'], node['splitter'])):
                    _, child_index = node['child{}'.format(i)]
                    if node['child{}'.format(i)][0] == 'node':
                        nodes[child_index]['hull'] = sub
                        # traverse_node(nodes[child_index])
                        if child_index >= node['id']:
                            print('error')
                            print(node)
                            sys.exit(0)
                    else:
                        subsectors[child_index]['hull'] = sub

            # traverse_node(nodes[-1])
            for subsector in subsectors:
                if 'hull' not in subsector: continue
                hull = subsector['hull']
                s = subsector['start_seg']
                n = subsector['numsegs']
                for seg in segs[s:s + n]:
                    vs = [tuple([vertexes[vertex_index][k + '_coord'] for k in ['X', 'Y']])
                          for vertex_index in [seg['vertex_' + k] for k in ['start', 'end']]]
                    segline = make_line(vs)
                    hull, _ = split_polygon(hull, segline)
                subsector['hull'] = hull
                # split_polygon(hull, segline)

            sf = 2000
            aspect_ratio = (bb[1][1] - bb[0][1]) / (bb[1][0] - bb[0][0])
            if aspect_ratio > 1:
                scale_factor = [sf / aspect_ratio, sf]
            else:
                scale_factor = [sf, sf * aspect_ratio]

            palette = load_palette(wad, directory_map)
            colormaps = load_colormaps(wad, directory_map)

            if False:
                # Make svg.
                dwg = svgwrite.Drawing('out.svg', profile='tiny')
                PADDING = 20
                dwg.viewbox(minx=0, miny=0, width=scale_factor[0] + 2 * PADDING, height=scale_factor[1] + 2 * PADDING)

                sgrp = dwg.g()
                sgrp.fill(opacity=0)
                sgrp.translate(tx=PADDING, ty=PADDING)
                sgrp.translate(tx=0, ty=1)
                sgrp.scale(sx=scale_factor[0], sy=scale_factor[1])
                sgrp.translate(tx=0, ty=1)
                sgrp.scale(sx=1 / (bb[1][0] - bb[0][0]), sy=-1 / (bb[1][1] - bb[0][1]))
                sgrp.translate(tx=-bb[0][0], ty=-bb[0][1])

                grp = dwg.g()
                grp.stroke(color='gray', width=5)
                for subsector in subsectors:
                    if 'hull' not in subsector: continue
                    if len(subsector['hull']) == 0: continue
                    vs = [[float(c) for c in v] for v in subsector['hull']]
                    # e = dwg.polygon(points=vs, fill=svgwrite.rgb(*sector_colors[poly.get_aux()['sector']['aux']['id']]))
                    sector = sectors[subsector['sector_ref']]
                    e = dwg.polygon(points=vs)
                    e.fill(color=svgwrite.rgb(*sector['floorcolor']))

                    grp.add(e)
                sgrp.add(grp)

                dwg.add(sgrp)
                dwg.save()

            # Get PNAMES
            wad.seek(directory_map['PNAMES']['filepos'])
            nummappatches, = struct.unpack('<I', wad.read(4))
            pnames = []
            for _ in range(nummappatches):
                pname = parse_string8(wad.read(8))
                pnames += [pname]

            # Load texture data.
            textures = {}
            textures['-'] = {'width': 1, 'height': 1}
            for tdname in ['TEXTURE{}'.format(k) for k in [1, 2]]:
                if tdname not in directory_map: continue
                tdoffset = directory_map[tdname]['filepos']
                wad.seek(tdoffset)
                numtextures, = struct.unpack('<I', wad.read(4))
                offsets = list(struct.unpack('<{}I'.format(numtextures), wad.read(4 * numtextures)))
                for offset in offsets:
                    wad.seek(tdoffset + offset)
                    if wadtype == 'strife':
                        texture_name, _, width, height, patchcount = struct.unpack('<8sIhhh', wad.read(18))
                    else:
                        texture_name, _, width, height, _, patchcount = struct.unpack('<8sIhhIh', wad.read(22))
                    texture_name = parse_string8(texture_name)
                    texture = {
                        'name': texture_name,
                        'width': width,
                        'height': height,
                        'texheight': least_twoexp(height),
                        'patches': []}
                    for _ in range(patchcount):
                        if wadtype == 'strife':
                            originx, originy, patch = struct.unpack('<3h', wad.read(6))
                        else:
                            originx, originy, patch, _, _ = struct.unpack('<5h', wad.read(10))
                        texture['patches'] += [{
                            'origin': (originx, originy),
                            'name': pnames[patch]}]

                    textures[texture_name] = texture

            # Group subsectors by texture.
            flat_groups = {}
            texture_groups = {}

            def texture_group_insert(groups, pic, item):
                if pic not in groups: groups[pic] = {'items': []}
                groups[pic]['items'] += [item['id']]

            for subsector in subsectors:
                sector = sectors[subsector['sector_ref']]
                texture_group_insert(flat_groups, sector['ceilingpic'], subsector)
                texture_group_insert(flat_groups, sector['floorpic'], subsector)

            for seg in segs:
                linedef = linedefs[seg['line_num']]
                sidedef_index = linedef['sidedef_' + ('right' if seg['segside'] == 0 else 'left')]
                sidedef = sidedefs[sidedef_index]
                texture_group_insert(texture_groups, sidedef['uppertexture'], seg)
                texture_group_insert(texture_groups, sidedef['middletexture'], seg)
                texture_group_insert(texture_groups, sidedef['lowertexture'], seg)

            for name in texture_groups.keys():
                if name == '-': continue
                texture_groups[name]['width'] = 1
                texture_groups[name]['texheight'] = 1
                texture_groups[name]['height'] = 1
                texture_groups[name]['picture'] = read_texture(wad, directory_map, textures[name.upper()])

            for name in flat_groups.keys():
                if name == '-': continue
                flat_groups[name]['picture'] = load_flat(wad, directory_map[name])

            sky_pictures = {}
            for k in ['{}SKY{}'.format(s, i + 1) for i in range(4) for s in ['', 'R']]:
                if k not in directory_map: continue
                data = [None] * 256 * 128
                load_picture(data, wad, directory_map[k], (256, 128))
                sky_pictures[k] = {'width': 256, 'height': 256, 'texheight': 256, 'data': data + data}
            if wadtype == 'strife':
                for k in ['SKYMNT0{}'.format(i + 1) for i in range(2)]:
                    sky_pictures[k] = read_texture(wad, directory_map, textures[k.upper()])
            if False:
                # for name in textures.keys():
                for name in texture_groups.keys():
                    if name == '-': continue
                    # tex = read_texture(wad, directory_map, textures[name.upper()])
                    write_texture(texture_normal(texture_groups[name]['picture']), name)
                sys.exit(0)

            # pprint.pprint(bsp.root())
        # pprint.pprint(directory[start_index:end_index])
        # pprint.pprint(directory[start_index:end_index])
        # pprint.pprint(directory[start_index:end_index])

    def traverse_bsp(node, position):
        side_index = (normalize_integer(line_distance(node['splitter'], position)) + 1) // 2
        child_type, child_index = node['child{}'.format(side_index)]
        if child_type == 'node':
            return traverse_bsp(nodes[child_index], position)
        else:
            return subsectors[child_index]

    player_start = [0, 0]
    for thing in things:
        if thing['type'] == 1:
            player_start = [thing[k + '_position'] for k in ['x', 'y']]
    player_start += [sectors[traverse_bsp(nodes[-1], player_start)['sector_ref']]['floorheight'] + 56 // 2]

    # Play the level.
    pygame.init()

    # (screenx, screeny)=(1280, 720)
    if STEREOSCOPIC:
        # (screenx, screeny)=(800, 600)
        (screenx, screeny) = (1200, 600)
    else:
        (screenx, screeny) = (1600, 900)

    pygame.display.set_mode((screenx, screeny), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.gl_set_attribute(pygame.GL_STEREO, 1)

    print(pygame.display.gl_get_attribute(pygame.GL_STEREO))

    # gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    gl.glMatrixMode(gl.GL_PROJECTION)

    def perspective(fov, aspect, near, far):
        t = near * math.tan(fov * math.pi / 360)
        r = t * aspect
        gl.glFrustum(-r, r, -t, t, near, far)

    perspective(90, 1.0 * screenx / screeny, 0.001, 10000)

    gl.glMatrixMode(gl.GL_MODELVIEW)
    player = Player(player_start, math.pi / 2)
    # player = Player()
    done = False
    gl.glFrontFace(gl.GL_CW)
    # gl.glFrontFace(gl.GL_CCW)
    gl.glEnable(gl.GL_CULL_FACE)
    # gl.glDisable(gl.GL_CULL_FACE)
    gl.glEnable(gl.GL_DEPTH_TEST)
    # gl.glEnable(gl.GL_BLEND)
    pygame.key.set_repeat(10, 5)

    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
    gl.glEnableClientState(gl.GL_INDEX_ARRAY)

    # for sector in sectors:
    #    print(sector['lightlevel'])
    # for subsector in subsectors:
    #    pprint.pprint(subsector)

    def produce_wall(seg, lower, upper, texture, sector, level=0, two_sided=True):
        vs = [[vertexes[vertex_index][k + '_coord'] for k in ['X', 'Y']]
              for vertex_index in [seg['vertex_' + k] for k in ['start', 'end']]]

        v3 = [vs[0] + [upper],
              vs[1] + [upper],
              vs[1] + [lower],
              vs[0] + [lower]]
        # t2 = [[0, 0], [1, 0], [1, 1], [0, 1]]
        wall_length = np.linalg.norm(np.array(vs[0]) - np.array(vs[1]))
        wall_height = upper - lower
        # print(wall_length)
        # print(texture)
        linedef = linedefs[seg['line_num']]
        sidedef_index = linedef['sidedef_' + ('right' if seg['segside'] == 0 else 'left')]
        sidedef = sidedefs[sidedef_index]
        ceiling = sectors[sidedef['sector_ref']]['ceilingheight']
        floor = sectors[sidedef['sector_ref']]['floorheight']
        # print(sidedef, seg)
        # print(seg['segoffset']+sidedef['xoffset'])
        # offset = np.array([seg['segoffset']+sidedef['xoffset'], sidedef['yoffset'], 0], dtype=np.float32)

        if level > 0:
            if (linedef['flags'] & 0x18) == 0:
                t = wall_height + texture['height'] ** 2 / texture['texheight']
                offset = np.array([seg['segoffset'] + sidedef['xoffset'], sidedef['yoffset'] - t, 0], dtype=np.float32)
            else:
                offset = np.array([seg['segoffset'] + sidedef['xoffset'], sidedef['yoffset'], 0], dtype=np.float32)
        elif level < 0:
            # offset = np.array([seg['segoffset']+sidedef['xoffset'], -upper+ceiling-sidedef['yoffset'], 0], dtype=np.float32)
            if (linedef['flags'] & 0x10) > 0:
                voffset = floor - ceiling
                t = wall_height + texture['height'] ** 2 / texture['texheight']
                offset = np.array([seg['segoffset'] + sidedef['xoffset'], sidedef['yoffset'] - voffset - t, 0],
                                  dtype=np.float32)
            else:
                offset = np.array([seg['segoffset'] + sidedef['xoffset'], sidedef['yoffset'], 0], dtype=np.float32)
        else:
            # TODO: Correct middle textures for two sided linedefs.
            if (linedef['flags'] & 0x10) > 0:
                t = wall_height + texture['height'] ** 2 / texture['texheight']
                offset = np.array([seg['segoffset'] + sidedef['xoffset'], sidedef['yoffset'] - t, 0], dtype=np.float32)
            else:
                offset = np.array([seg['segoffset'] + sidedef['xoffset'], sidedef['yoffset'], 0], dtype=np.float32)

        offset /= np.array([texture['width'], texture['height'], 1], dtype=np.float32)

        # yoffset+upper sector floor height = 0 (mod texture height)
        # print(offset)
        # print(sidedef)
        t2 = [None] * 4
        t2[0] = [0, 0, 0]
        t2[1] = [0, 0, 0]
        t2[2] = [0, 0, 0]
        t2[3] = [0, 0, 0]
        t2[1][0] = wall_length / texture['width']
        t2[2][0] = wall_length / texture['width']
        t2[2][1] = wall_height / texture['texheight']
        t2[3][1] = wall_height / texture['texheight']
        v3 = [np.array(v, dtype=np.float32) for v in v3]
        t2 = [np.array(t + offset, dtype=np.float32) for t in t2]
        # Bli78
        normal = normalize(np.cross(v3[3] - v3[0], v3[1] - v3[0]))
        # normal = (np.cross(v3[3]-v3[0], v3[1]-v3[0]))
        # normal /= math.sqrt(np.linalg.norm(normal))
        tangent = v3[1] - v3[0]
        bitangent = v3[3] - v3[0]
        tangent_lengths = np.array([np.linalg.norm(v) for v in [tangent, bitangent, normal]], dtype=np.float32)
        for v, n in zip([tangent, bitangent, normal], tangent_lengths):
            if n != 0:
                v /= n

        lightlevel = (sector['lightlevel'] >> 3) & 31
        for vertex, texcoord in zip(v3, t2):
            texcoord[2] = lightlevel
            yield np.append(vertex, [normal, texcoord, tangent, bitangent, tangent_lengths])

    def produce_flat(hull, height, sector, is_ceiling=False):
        normal = np.array([0, 0, -1 if is_ceiling else 1], dtype=np.float32)
        # tangent = normalize(np.array(hull[-1 if is_ceiling else 1]+[height], dtype=np.float32) - np.array(hull[0]+[height], dtype=np.float32))
        tangent = np.array([64, 0, 0], dtype=np.float32)
        bitangent = np.array([0, -64, 0], dtype=np.float32)

        tangent_lengths = np.array([np.linalg.norm(v) for v in [tangent, bitangent, normal]], dtype=np.float32)
        for v, n in zip([tangent, bitangent, normal], tangent_lengths):
            if n != 0:
                v /= n

        lightlevel = (sector['lightlevel'] >> 3) & 31
        for v in (reversed(hull) if is_ceiling else hull):
            vertex = np.array(v + [height], dtype=np.float32)
            texcoord = np.array([1, -1, 0], dtype=np.float32) * np.array(v[:2] + [0], dtype=np.float32) / 64
            texcoord[2] = lightlevel
            yield np.append(vertex, [normal, texcoord, tangent, bitangent, tangent_lengths])

    if wadtype == 'hexen':
        SKY_FLAT = 'F_SKY'
    elif wadtype == 'strife':
        SKY_FLAT = 'F_SKY001'
    else:
        SKY_FLAT = 'F_SKY1'
    vertex_array = []
    index_array = []
    starting_index = 0
    index_count = 0
    for picname, group in flat_groups.items():
        group['starting index'] = index_count
        group['ending index'] = index_count
        group['count'] = 0

        if picname == '-': continue
        if picname == SKY_FLAT: continue

        triangle_count = 0
        group['starting index'] = index_count
        for subsector in [subsectors[i] for i in group['items']]:
            if len(subsector['hull']) < 3: continue
            sector = sectors[subsector['sector_ref']]

            # Make floors.
            if sector['floorpic'] == picname:
                vertex_array += [list(produce_flat(subsector['hull'], sector['floorheight'], sector))]
                index_array += [np.array(
                    list(map(lambda i: i + starting_index,
                             sum([[0, i, i + 1] for i in range(1, len(subsector['hull']) - 1)], []))),
                    dtype=np.int32)]
                starting_index += len(subsector['hull'])
                triangle_count += len(subsector['hull']) - 2
                index_count += (len(subsector['hull']) - 2) * 3
            # Make ceilings.
            if sector['ceilingpic'] == picname:
                vertex_array += [
                    list(produce_flat(subsector['hull'], sector['ceilingheight'], sector, is_ceiling=True))]
                index_array += [np.array(
                    list(map(lambda i: i + starting_index,
                             sum([[0, i, i + 1] for i in range(1, len(subsector['hull']) - 1)], []))),
                    dtype=np.int32)]
                starting_index += len(subsector['hull'])
                triangle_count += len(subsector['hull']) - 2
                index_count += (len(subsector['hull']) - 2) * 3
        group['ending index'] = index_count
        group['count'] = triangle_count

    for picname, group in texture_groups.items():
        group['starting index'] = index_count
        group['ending index'] = index_count
        group['count'] = 0

        if picname == '-': continue

        triangle_count = 0
        # Make walls.
        group['starting index'] = index_count
        for seg in [segs[i] for i in group['items']]:
            linedef = linedefs[seg['line_num']]
            sidedef_index = linedef['sidedef_' + ('right' if seg['segside'] == 0 else 'left')]
            other_sidedef_index = linedef['sidedef_' + ('left' if seg['segside'] == 0 else 'right')]
            sidedef = sidedefs[sidedef_index]
            sector = sectors[sidedef['sector_ref']]
            if other_sidedef_index is None:
                if picname == sidedef['middletexture']:
                    vertex_array += [list(
                        produce_wall(seg, sector['floorheight'], sector['ceilingheight'], textures[picname.upper()],
                                     sector, two_sided=False))]
                    index_array += [np.array(
                        list(map(lambda i: i + starting_index, sum([[0, i, i + 1] for i in range(1, 3)], []))),
                        dtype=np.int32)]
                    starting_index += 4
                    triangle_count += 2
                    index_count += 6
            else:
                other_sidedef = sidedefs[other_sidedef_index]
                other_sector = sectors[other_sidedef['sector_ref']]
                # Make upper wall.
                if picname == sidedef['uppertexture'] and sector['ceilingheight'] > other_sector['ceilingheight']:
                    if other_sector['ceilingpic'] != SKY_FLAT:
                        vertex_array += [list(produce_wall(seg, other_sector['ceilingheight'], sector['ceilingheight'],
                                                           textures[picname.upper()], sector, 1))]
                        index_array += [np.array(
                            list(map(lambda i: i + starting_index, sum([[0, i, i + 1] for i in range(1, 3)], []))),
                            dtype=np.int32)]
                        starting_index += 4
                        triangle_count += 2
                        index_count += 6
                # Make lower wall.
                if picname == sidedef['lowertexture'] and sector['floorheight'] < other_sector['floorheight']:
                    if other_sector['floorpic'] != SKY_FLAT:
                        vertex_array += [list(produce_wall(seg, sector['floorheight'], other_sector['floorheight'],
                                                           textures[picname.upper()], sector, -1))]
                        index_array += [np.array(
                            list(map(lambda i: i + starting_index, sum([[0, i, i + 1] for i in range(1, 3)], []))),
                            dtype=np.int32)]
                        starting_index += 4
                        triangle_count += 2
                        index_count += 6

                # Make middle wall.
                if picname == sidedef['middletexture'] and \
                        sector['floorheight'] < other_sector['ceilingheight'] and \
                        other_sector['floorheight'] < other_sector['ceilingheight'] and \
                        other_sector['floorheight'] < sector['ceilingheight']:
                    texture_height = min(128, other_sector['ceilingheight'] - other_sector['floorheight'])
                    vertex_array += [list(
                        produce_wall(seg, other_sector['floorheight'], other_sector['floorheight'] + texture_height,
                                     textures[picname.upper()], sector, 0))]
                    # vertex_array += [list(produce_wall(seg, other_sector['floorheight'], other_sector['ceilingheight'], textures[picname.upper()], 0))]
                    index_array += [np.array(
                        list(map(lambda i: i + starting_index, sum([[0, i, i + 1] for i in range(1, 3)], []))),
                        dtype=np.int32)]
                    starting_index += 4
                    triangle_count += 2
                    index_count += 6

        group['ending index'] = index_count
        group['count'] = triangle_count
    # Collect sky wall textures.
    sky_group = {}
    data = []

    if mapname[0] == 'E' and mapname[2] == 'M':
        sky_name = 'SKY{}'.format(int(mapname[1]))
    elif mapname[:3] == 'MAP':
        map_index = int(mapname[3:])
        sky_index = 1 if map_index <= 11 else 2 if map_index <= 20 else 3
        sky_name = 'RSKY{}'.format(sky_index)
    if sky_name in sky_pictures:
        sky_group['picture'] = sky_pictures[sky_name]
    else:
        sky_group['picture'] = sky_pictures[list(sky_pictures.keys())[0]]
    sky_group['flat starting index'] = index_count
    triangle_count = 0
    for subsector in subsectors:
        if len(subsector['hull']) < 3: continue
        sector = sectors[subsector['sector_ref']]

        # Make floors.
        if sector['floorpic'] == SKY_FLAT:
            vertex_array += [list(produce_flat(subsector['hull'], sector['floorheight'], sector))]
            index_array += [np.array(
                list(map(lambda i: i + starting_index,
                         sum([[0, i, i + 1] for i in range(1, len(subsector['hull']) - 1)], []))),
                dtype=np.int32)]
            starting_index += len(subsector['hull'])
            triangle_count += len(subsector['hull']) - 2
            index_count += (len(subsector['hull']) - 2) * 3
        # Make ceilings.
        if sector['ceilingpic'] == SKY_FLAT:
            vertex_array += [list(produce_flat(subsector['hull'], sector['ceilingheight'], sector, is_ceiling=True))]
            index_array += [np.array(
                list(map(lambda i: i + starting_index,
                         sum([[0, i, i + 1] for i in range(1, len(subsector['hull']) - 1)], []))),
                dtype=np.int32)]
            starting_index += len(subsector['hull'])
            triangle_count += len(subsector['hull']) - 2
            index_count += (len(subsector['hull']) - 2) * 3

    sky_group['flat ending index'] = index_count
    sky_group['flat count'] = triangle_count

    min_height = sectors[0]['floorheight']
    max_height = sectors[0]['ceilingheight']

    for sector in sectors[1:]:
        min_height = min(min_height, sector['floorheight'])
        max_height = max(max_height, sector['ceilingheight'])

    sky_group['wall starting index'] = index_count
    triangle_count = 0
    for seg in segs:
        linedef = linedefs[seg['line_num']]
        sidedef_index = linedef['sidedef_' + ('right' if seg['segside'] == 0 else 'left')]
        other_sidedef_index = linedef['sidedef_' + ('left' if seg['segside'] == 0 else 'right')]
        sidedef = sidedefs[sidedef_index]
        sector = sectors[sidedef['sector_ref']]
        if other_sidedef_index is not None:
            other_sidedef = sidedefs[other_sidedef_index]
            other_sector = sectors[other_sidedef['sector_ref']]
            if sector['floorheight'] < other_sector['floorheight'] and other_sector['floorpic'] == SKY_FLAT:
                vertex_array += [list(
                    produce_wall(seg, sector['floorheight'], other_sector['floorheight'], textures[picname.upper()],
                                 sector, -1))]
                # vertex_array += [list(produce_wall(seg, min_height, other_sector['floorheight'], textures[picname.upper()], -1))]
                index_array += [np.array(
                    list(map(lambda i: i + starting_index, sum([[0, i, i + 1] for i in range(1, 3)], []))),
                    dtype=np.int32)]
                starting_index += 4
                triangle_count += 2
                index_count += 6
            if other_sector['ceilingheight'] < sector['ceilingheight'] and other_sector['ceilingpic'] == SKY_FLAT:
                vertex_array += [list(
                    produce_wall(seg, other_sector['ceilingheight'], sector['ceilingheight'], textures[picname.upper()],
                                 sector, 1))]
                # vertex_array += [list(produce_wall(seg, other_sector['ceilingheight'], max_height, textures[picname.upper()], 1))]
                index_array += [np.array(
                    list(map(lambda i: i + starting_index, sum([[0, i, i + 1] for i in range(1, 3)], []))),
                    dtype=np.int32)]
                starting_index += 4
                triangle_count += 2
                index_count += 6
                index_count += 6

    sky_group['wall ending index'] = index_count
    sky_group['wall count'] = triangle_count

    # for picname, group in texture_groups.items():
    #    if picname=='-': continue
    #    print(picname, group['starting index'], group['ending index'], group['count'])

    vertex_array = np.concatenate(vertex_array)
    index_array = np.concatenate(index_array)
    # print(index_array)
    # print(vertex_array)

    vbo_vertices = vbo.VBO(vertex_array)
    vbo_indices = vbo.VBO(index_array, target=gl.GL_ELEMENT_ARRAY_BUFFER)

    vbo_vertices.bind()
    vbo_indices.bind()

    VERTEX_SHADER_CODE = """// GLSL
#version 130
in vec3 aPosition;
in vec3 aNormal;
in vec3 aTangent;
in vec3 aBitangent;
in vec2 aTexCoord;
out vec3 vNormal;
out vec4 vPosition;
out vec2 vTexCoord;
out vec3 vView;
out vec3 vLight;
//out mat3 tangent_basis;
out mat3 tangent_normbasis;
in vec3 v_tangent_lengths;
in float v_lightlevel;
flat out float lightlevel;
out vec3 tangent_lengths;
out float v_dist;
void main() {
    vTexCoord = aTexCoord;
    vNormal = normalize(gl_NormalMatrix * aNormal);
    vPosition = gl_ModelViewMatrix * vec4(aPosition, 1);
    gl_Position = gl_ProjectionMatrix * vPosition;
    v_dist = gl_Position.w;
    /* Maps from model space to tangent space when multiplied by a row vector. */
    tangent_normbasis[0] = aTangent;
    tangent_normbasis[1] = aBitangent;
    tangent_normbasis[2] = aNormal;
    tangent_normbasis = gl_NormalMatrix * tangent_normbasis;
    tangent_lengths = v_tangent_lengths;
    lightlevel = v_lightlevel;
}
"""

    FRAGMENT_SHADER_CODE = """// GLSL
#version 130
flat in float lightlevel;
in vec3 vNormal;
in vec4 vPosition;
in vec2 vTexCoord;
in mat3 tangent_basis;
in mat3 tangent_normbasis;
in vec3 vView;
in vec3 tangent_lengths;
in vec4 vProjPosition;
uniform sampler2D diffuseSampler;
uniform sampler2D normalSampler;
uniform sampler2D colormapSampler;
// The relief mapping scale and bias.
uniform vec2 scale_bias;
uniform vec2 texture_size;
// The distance to the fragment.
in float v_dist;
#define DIST_SCALE 64.0
#define LIGHT_SCALE 2.0
#define LIGHT_BIAS 1e-4
#define LINEAR_SEARCH_STEPS    256
#define BIN_ITER 10
#define USE_PARALLAX_MAPPING    0
#define USE_BINARY_SEARCH    0
#define USE_BINARY_REFINEMENT    1
#define USE_LINEAR_SEARCH    1
#define USE_SPHERE_BUMP        0
vec4 sphere_normal(vec2 uv) {
    vec2 xy = mod(uv, 1)*2-1;
    float t = sqrt(max(1-dot(xy, xy), 0));
    vec4 v;
    if(t!=0) {
        v = vec4(xy, t, t);
    } else {
        v = vec4(0, 0, 1, t);
    }
    return v;
}
vec3 line_sphere(vec3 c, float r, vec3 o, vec3 l) {
    vec3 oc = o-c;
    float t = dot(l, oc);
    float disc = t*t - dot(oc, oc) + r*r;
    float d;
    if(disc>=0) {
        float d0 = -t + disc;
        float d1 = -t - disc;
        d= min(d0, d1);
    } else {
        d = 0;
    }
    return o+d*l;
}
float height_map(vec2 uv) {
#if USE_SPHERE_BUMP
    float h = sphere_normal(uv).a;
#else
    float h = texture(normalSampler, uv).a;
#endif
    return dot(vec2(h, 1), scale_bias);
}
/* Interpolate view between the top height level and the bottom height level respectively. */
vec3 uvt_lerp(float t, vec2 uv, vec3 view) {
    vec3 uv0 = vec3(uv, 0);
    //vec3 uv_min = uv0 - scale_bias.y * view.xy / view.z;
    return uv0 + dot(scale_bias, vec2(1-t, 1)) * vec3(view.xy / view.z, 1);
}
vec2 binsearch(vec2 uv, vec3 view, float a, float b) {
    vec3 uv_i;
    // f(tmax) is below the curve whereas f(tmin) is above the curve.
    float tmax=b, tmin = a;
    for(int i=0;i<BIN_ITER;i++) {
        float t = (tmax+tmin)/2;
        uv_i = uvt_lerp(t, uv, view);
        float h = height_map(uv_i.xy);
        if(uv_i.z > h) {
            tmin = t;
        } else {
            tmax = t;
        }
    }
    return uv_i.xy;
}
vec2 linsearch(vec2 uv, vec3 view, int steps) {
    for(int i=0;i<steps;i++) {
        vec3 uv_i = uvt_lerp(float(i)/steps, uv, view);
        float h = height_map(uv_i.xy);
        if(h >= uv_i.z) {
#if USE_BINARY_REFINEMENT
            return binsearch(uv, view, float(i-1)/steps, float(i)/steps);
#else
            return uv_i.xy;
#endif
        }
    }
    return uvt_lerp(1, uv, view).xy;
}
vec4 retro_color(float lightlevel, float diffuseIndex) {
    float lightlevel_scaled = lightlevel / 31.0;
    float dist_term = min(1.0, 1.0 - DIST_SCALE / (v_dist + DIST_SCALE));
    float light_discrete = min(lightlevel_scaled, lightlevel_scaled * LIGHT_SCALE - dist_term);
    light_discrete = clamp(1.0 - light_discrete, LIGHT_BIAS, 1.0 - LIGHT_BIAS);
    light_discrete *= 32/34.0;
    return texture(colormapSampler, vec2(diffuseIndex, light_discrete));
}
vec4 bilinearInterpolate(sampler2D samp, vec2 uv, float lightlevel) {
    vec2 uv00 = floor(uv*texture_size) / texture_size;
    vec4 p00 = retro_color(lightlevel, texture(samp, uv00).r);
    vec2 uv10 = floor(uv*texture_size + vec2(1,0)) / texture_size;
    vec4 p10 = retro_color(lightlevel, texture(samp, uv10).r);
    vec2 uv01 = floor(uv*texture_size + vec2(0,1)) / texture_size;
    vec4 p01 = retro_color(lightlevel, texture(samp, uv01).r);
    vec2 uv11 = floor(uv*texture_size + vec2(1,1)) / texture_size;
    vec4 p11 = retro_color(lightlevel, texture(samp, uv11).r);
    return 
    mix(
    mix(p00,p01, mod(uv.y*texture_size.y,1.0) ),
    mix(p10,p11, mod(uv.y*texture_size.y,1.0) ),
    mod(uv.x*texture_size.x,1.0)
    )
    ;
}
void main() {
#if USE_SPHERE_BUMP
    vec4 nm = sphere_normal(vTexCoord);
#else
    vec4 nm = texture(normalSampler, vTexCoord)*2-1;
#endif
    float attenuation = 0.0;
    /* The light vector is the normalized view vector pointing at the viewer.
       It coincides with surface normals. */
    vec3 light = -normalize(vPosition.xyz*tangent_normbasis);
    vec2 uv = vTexCoord;
    /* The view vector is in the texture space (the tangent space scaled to
       texture coordinates. It points away from the viewer. */
    //vec3 view = vPosition.xyz*tangent_normbasis/tangent_lengths;
    vec3 view = vPosition.xyz * tangent_normbasis/tangent_lengths;
    /* Perform parallax mapping. */
#if USE_PARALLAX_MAPPING
    float h = dot(vec2(nm.a, 1), scale_bias);
    uv += h*view.xy/view.z;
#endif
#if USE_LINEAR_SEARCH
    uv = linsearch(uv, view, LINEAR_SEARCH_STEPS);
#endif
#if USE_BINARY_SEARCH
    uv = binsearch(uv, view, 0, 1);
#endif
    nm = texture(normalSampler, uv)*2-1;
#if 0
    vec4 diffuseIndex = texture(diffuseSampler, uv);
    vec4 color = retro_color(lightlevel, diffuseIndex.r);
    if(diffuseIndex.a < 0.5) {
        discard;
    }
#else
    vec4 diffuseIndex = texture(diffuseSampler, uv);
    if(diffuseIndex.a < 0.5) {
        discard;
    }
    vec4 color = bilinearInterpolate(diffuseSampler, uv, lightlevel);
#endif
    //color.a = 1.0;
    color.rgb += 1e-2;
    attenuation += max(dot(nm.rgb, light), 0);
    //attenuation = max(attenuation,0.75);
    gl_FragColor = color*attenuation;
    //gl_FragColor = color;
}
"""

    diffuse_texture = gl.glGenTextures(1)
    normal_texture = gl.glGenTextures(1)
    sky_texture = gl.glGenTextures(1)

    # Create the paletted textures.
    gl.glActiveTexture(gl.GL_TEXTURE2)
    colormap_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, colormap_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 256, 34, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                    np.concatenate(np.transpose(colormaps, (1, 0, 2))))

    def load_texture(picture, light_level, is_flat=True):
        if 'colors' not in picture:
            data = picture['data']
            texture_data = np.zeros((picture['width'], picture['height'], 4), dtype=np.uint8)
            texture_indexes = np.zeros((picture['width'], picture['height'], 4), dtype=np.uint8)
            # colormap = colormaps[(255-light_level) // 8]
            # for c in data:
            for x, y in itertools.product(range(picture['width']), range(picture['height'])):
                c = data[x + y * picture['width']]
                if c is not None:
                    color = list(palette[c])
                else:
                    color = [0, 0, 0, 0]
                texture_data[x, y, :] = color
                if c is not None:
                    texture_indexes[x, y, 0] = c
                    texture_indexes[x, y, 3] = 255
                else:
                    texture_indexes[x, y, 3] = 0

            picture['colors pixels'] = texture_data
            picture['colors'] = np.concatenate(np.transpose(texture_indexes, (1, 0, 2)))
        if 'normals' not in picture:
            BUMP_STRENGTH = 2
            # intensities = np.average(np.array(picture['colors pixels'], dtype=np.float32), axis=2)/255
            # intensities = np.linalg.norm(ndimage.filters.sobel(np.array(picture['colors pixels'], dtype=np.float32)/255), axis=2)
            # intensities = np.average(ndimage.filters.laplace(np.array(picture['colors pixels'], dtype=np.float32)/255, mode='wrap'), axis=2)
            intensities = np.linalg.norm(
                ndimage.filters.laplace(np.array(picture['colors pixels'], dtype=np.float32) / 255, mode='wrap'),
                axis=2)

            # intensities = ndimage.gaussian_filter(intensities, sigma=1.5, mode='wrap')
            intensities = ndimage.gaussian_filter(intensities, sigma=1.0, mode='wrap')

            zs = np.zeros((picture['width'], picture['height']), dtype=np.float32)
            if False:
                wh = [picture['width'], picture['height']]
                # intensities = np.max(1-np.sum(((np.transpose(np.mgrid[:wh[0], :wh[1]], (1, 2, 0)) - wh) / wh)**2, axis=2), zs)
                intensities = 1 - np.sqrt(
                    np.maximum(1 - np.sum(
                        ((np.transpose(np.mgrid[:wh[0], :wh[1]], (1, 2, 0)) - np.array(wh, dtype=np.float32) / 2)
                         / (np.array(wh, dtype=np.float32) / 2)) ** 2, axis=2), zs)
                )
                print(intensities)
                # print((1-np.sum(((np.transpose(np.mgrid[:wh[0], :wh[1]], (1, 2, 0)) - wh) / wh)**2, axis=2)).shape, zs.shape)

            def rescale(x):
                x -= np.amin(x)
                mx = np.amax(x)
                if mx != 0:
                    x /= mx
                return x

            intensities = 1 - rescale(intensities)

            nd = np.transpose(
                [ndimage.filters.sobel(intensities, axis=axis, mode='wrap') for axis in range(2)] + [zs] * 2, (1, 2, 0))
            nd[:, :, 2] = -1
            nd[:, :, 3] = intensities

            ndn = -np.linalg.norm(nd[:, :, :3], axis=2)
            nd[:, :, 0] /= ndn
            nd[:, :, 1] /= ndn
            nd[:, :, 2] /= ndn
            nd = (nd + 1) / 2

            normal_data = nd

            picture['normals pixels'] = normal_data
            picture['normals'] = np.concatenate(np.transpose(normal_data, (1, 0, 2)))

        if 'normals' not in picture and False:

            def apply_kernel(x, y, px, width, height, kernel):
                s = 0
                for i in range(3):
                    xi = i + x - 1
                    if not (0 <= xi and xi < width): continue
                    for j in range(3):
                        yj = y + j - 1
                        if not (0 <= yj and yj < height): continue
                        s += kernel[i + j * 3] * px[xi + yj * width]
                return s

            XKERNEL = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
            YKERNEL = [1, 2, 1, 0, 0, 0, -1, -2, -1]
            ZSTRENGTH = 1
            intensities = [0] * picture['width'] * picture['height']

            for x in range(picture['width']):
                for y in range(picture['height']):
                    idx = x + y * picture['width']
                    c = sum(picture['colors'][idx * 4:(idx + 1) * 4 - 1]) / 255.0
                    intensities[idx] = c
            normal_data = []
            for x in range(picture['width']):
                for y in range(picture['width']):
                    # idx = x+y*picture['width']
                    # c = sum(picture['colors'][idx*4:(idx+1)*4-1])/255.0
                    # v = np.array(
                    #  [apply_kernel(x, y, intensities, picture['width'], picture['height'], k) for k in [XKERNEL, YKERNEL]]+[1/ZSTRENGTH]
                    #  , dtype=np.float32)
                    # v /= np.linalg.norm(v)
                    # print(v)
                    normal_data += [v]

            normal_data = np.concatenate(normal_data)
            picture['normals'] = normal_data
        width, height = picture['width'], picture['height']
        texture_data = picture['colors']
        normal_data = picture['normals']

        gl.glUniform2f(gl.glGetUniformLocation(shader, b'texture_size'), width, height)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, diffuse_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        texture_data)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, normal_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, normal_data)

        return texture, width, height

    def load_sky(picture):
        if 'colors' not in picture:
            data = picture['data']
            texture_data = np.zeros((picture['width'], picture['height'], 4), dtype=np.uint8)
            texture_indexes = np.zeros((picture['width'], picture['height'], 4), dtype=np.uint8)

            for x, y in itertools.product(range(picture['width']), range(picture['height'])):
                c = data[x + y * picture['width']]
                if c is not None:
                    color = list(palette[c])
                else:
                    color = [0, 0, 0, 0]
                texture_data[x, y, :] = color
                if c is not None:
                    texture_indexes[x, y, 0] = c
                    texture_indexes[x, y, 3] = 255
                else:
                    texture_indexes[x, y, 3] = 0

            texture_indexes = np.concatenate(np.transpose(texture_indexes, (1, 0, 2)))
            texture_data = np.concatenate(texture_data)
            picture['colors'] = texture_data
            picture['indexes'] = texture_indexes
            print(texture_data)
        width, height = picture['width'], picture['height']
        texture_data = picture['colors']
        texture_indexes = picture['indexes']

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, sky_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                        texture_indexes)

        # empty_data = np.array([0, 255, 255], dtype=np.uint8)
        # gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, gl.GL_RGB, 1, 1, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, empty_data)
        # gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 0, gl.GL_RGB, 1, 1, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, empty_data)

    # load_texture(textures['SKY1'], 255)
    # load_texture(flats['F_SKY1'], 255)

    vertex_shader = shaders.compileShader(VERTEX_SHADER_CODE, gl.GL_VERTEX_SHADER)
    fragment_shader = shaders.compileShader(FRAGMENT_SHADER_CODE, gl.GL_FRAGMENT_SHADER)

    shader = shaders.compileProgram(vertex_shader, fragment_shader)

    SKY_VERTEX_SHADER_CODE = """// GLSL
#version 130
in vec3 aPosition;
out vec3 vCoords;
void main() {
    vec4 position = gl_ModelViewMatrix * vec4(aPosition, 1);
    vCoords = vec3((position - vec4(0, 0, 0, 1))*gl_ModelViewMatrix);
    gl_Position = gl_ProjectionMatrix * position;
}"""

    SKY_FRAGMENT_SHADER_CODE = """// GLSL
#version 130
#define PI 3.141592653589793238
in vec3 vCoords;
uniform sampler2D cubeSampler;
uniform sampler2D colormapSampler;
void main() {
       vec3 v = (vCoords * vec3(1, 1, -1)) / length(vCoords.xy);
       vec3 t = abs(v);
       vec2 s = vec2(0, 0);

       s.t = v.z;
       if(t.x <= t.y) {
               s.s = v.x/t.y;
       } else {
               s.s = v.y/t.x;
       }

       s /= 2;
        vec4 diffuseIndex = texture(cubeSampler, mod(s, 1));
        vec4 color = texture(colormapSampler, vec2(diffuseIndex.r, 0.5));
       gl_FragColor = color;
}
"""

    def draw_scene():
        if False:
            # dd = demodata.demodata[gametic % len(demodata.demodata)]
            dd = demodata.demodata[int((time.clock() - timebegin) / INTERVAL) % len(demodata.demodata)]

            player.set_position(dd['pos'])
            player.move_up(40)
            player.set_angle(dd['angle'])

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN and pygame.key.get_mods() & pygame.KMOD_LALT:
                scale = 20 if pygame.key.get_mods() & pygame.KMOD_LSHIFT else 1
                if event.key == pygame.K_LEFT:
                    player.move_left(STRAFE_DISTANCE * scale)
                elif event.key == pygame.K_RIGHT:
                    player.move_left(-STRAFE_DISTANCE * scale)
                elif event.key == pygame.K_UP:
                    player.move_up(STRAFE_DISTANCE * scale)
                elif event.key == pygame.K_DOWN:
                    player.move_up(-STRAFE_DISTANCE * scale)
            elif event.type == pygame.KEYDOWN and pygame.key.get_mods() & pygame.KMOD_LCTRL:
                if event.key == pygame.K_UP:
                    player.look_up(TURN_ANGLE)
                elif event.key == pygame.K_DOWN:
                    player.look_up(-TURN_ANGLE)
            elif event.type == pygame.KEYDOWN:
                scale = 20 if pygame.key.get_mods() & pygame.KMOD_LSHIFT else 1
                if event.key == pygame.K_LEFT:
                    player.turn_left(TURN_ANGLE)
                elif event.key == pygame.K_RIGHT:
                    player.turn_left(-TURN_ANGLE)
                elif event.key == pygame.K_UP:
                    player.move_forward(FORWARD_DISTANCE * scale)
                elif event.key == pygame.K_DOWN:
                    player.move_forward(-FORWARD_DISTANCE * scale)
                elif event.key == pygame.K_q:
                    done = True
            # elif event.type==pygame.JOYAXISMOTION:
            #    print(joystick.get_axis(0), joystick.get_axis(1))
            # player.move_forward(FORWARD_DISTANCE * -joystick.get_axis(1))
            # player.move_left(STRAFE_DISTANCE * -joystick.get_axis(0))

        def joystick_deadzone(x):
            return x if abs(x) > JOYSTICK_DEADZONE else 0

        if joystick is not None:
            joystick_boost = math.pow(2, joystick_deadzone(joystick.get_axis(2) - joystick.get_axis(5)) * 3)
            player.move_forward(
                FORWARD_DISTANCE * JOYSTICK_SCALE * -joystick_deadzone(joystick.get_axis(1)) * joystick_boost)
            player.move_left(
                STRAFE_DISTANCE * JOYSTICK_SCALE * -joystick_deadzone(joystick.get_axis(0)) * joystick_boost)
            player.turn_left(TURN_ANGLE * JOYSTICK_TURN_SCALE * -joystick_deadzone(joystick.get_axis(3)))
            player.look_up(TURN_ANGLE * JOYSTICK_TURN_SCALE * -joystick_deadzone(joystick.get_axis(4)))

        node_type, node = ('node', nodes[-1])
        while node_type == 'node':
            side = line_distance(node['splitter'], player.get_position()[:2])
            # print(node, side)
            if side <= 0:
                node_type, node_index = node['child0']
            else:
                node_type, node_index = node['child1']
            if node_type == 'node':
                node = nodes[node_index]
            else:
                node = subsectors[node_index]
            # print(node['id'], node_type)
        # pprint.pprint(sectors[node['sector_ref']]['floorheight']+40)
        if joystick is not None and joystick.get_button(0):
            player.set_elevation(sectors[node['sector_ref']]['floorheight'] + 40)

        if joystick is not None and joystick.get_button(3):
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glPolygonMode(gl.GL_FRONT, gl.GL_FILL)

        # stride = 6*4+3*4+2*3*4 + 3*4 + 4
        stride = 6 * 4 + 3 * 4 + 2 * 3 * 4 + 3 * 4
        # Render the sky.
        shaders.glUseProgram(sky_shader)

        shaders.glUseProgram(sky_shader)
        sky_position_location = gl.glGetAttribLocation(sky_shader, b'aPosition')
        if sky_position_location >= 0:
            gl.glEnableVertexAttribArray(sky_position_location)
            gl.glVertexAttribPointer(sky_position_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices)

        texcoord_location = gl.glGetAttribLocation(sky_shader, b'aTexCoord')
        if texcoord_location >= 0:
            gl.glEnableVertexAttribArray(texcoord_location)
            gl.glVertexAttribPointer(texcoord_location, 2, gl.GL_FLOAT, False, stride, vbo_vertices + 6 * 4)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1i(gl.glGetUniformLocation(sky_shader, b'cubeSampler'), 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, sky_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glUniform1i(gl.glGetUniformLocation(sky_shader, b'colormapSampler'), 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, colormap_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        # gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, sky_texture)
        # gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        # gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        # gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        # gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        # gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        load_sky(sky_group['picture'])
        gl.glDrawElements(gl.GL_TRIANGLES, sky_group['flat count'] * 3, gl.GL_UNSIGNED_INT,
                          vbo_indices + sky_group['flat starting index'] * 4)
        # gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glDrawElements(gl.GL_TRIANGLES, sky_group['wall count'] * 3, gl.GL_UNSIGNED_INT,
                          vbo_indices + sky_group['wall starting index'] * 4)

        # gl.glDrawElements(gl.GL_LINES, len(index_array), gl.GL_UNSIGNED_INT, None)
        #    gl.glDrawRangeElements(gl.GL_TRIANGLES, 
        #            group['starting index'], 
        #            group['ending index'], 
        #            group['count'], gl.GL_UNSIGNED_INT, None)

        # gl.glDrawElements(gl.GL_TRIANGLES, len(index_array), gl.GL_UNSIGNED_INT, vbo_indices+12)

        shaders.glUseProgram(shader)

        # gl.glVertexPointer(3, gl.GL_FLOAT, 0, None)
        # gl.glVertexPointerf(vbo_positions)
        shaders.glUseProgram(shader)
        position_location = gl.glGetAttribLocation(shader, b'aPosition')
        if position_location >= 0:
            gl.glEnableVertexAttribArray(position_location)
            gl.glVertexAttribPointer(position_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices)

        normal_location = gl.glGetAttribLocation(shader, b'aNormal')
        if normal_location >= 0:
            gl.glEnableVertexAttribArray(normal_location)
            gl.glVertexAttribPointer(normal_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices + 3 * 4)

        tangent_location = gl.glGetAttribLocation(shader, b'aTangent')
        if tangent_location >= 0:
            gl.glEnableVertexAttribArray(tangent_location)
            gl.glVertexAttribPointer(tangent_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices + 9 * 4)

        bitangent_location = gl.glGetAttribLocation(shader, b'aBitangent')
        if bitangent_location >= 0:
            gl.glEnableVertexAttribArray(bitangent_location)
            gl.glVertexAttribPointer(bitangent_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices + 12 * 4)

        texcoord_location = gl.glGetAttribLocation(shader, b'aTexCoord')
        if texcoord_location >= 0:
            gl.glEnableVertexAttribArray(texcoord_location)
            gl.glVertexAttribPointer(texcoord_location, 2, gl.GL_FLOAT, False, stride, vbo_vertices + 6 * 4)

        tangent_lengths_location = gl.glGetAttribLocation(shader, b'v_tangent_lengths')
        if tangent_lengths_location >= 0:
            gl.glEnableVertexAttribArray(tangent_lengths_location)
            gl.glVertexAttribPointer(tangent_lengths_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices + 15 * 4)

        lightlevel_location = gl.glGetAttribLocation(shader, b'v_lightlevel')
        if lightlevel_location >= 0:
            gl.glEnableVertexAttribArray(lightlevel_location)
            gl.glVertexAttribPointer(lightlevel_location, 1, gl.GL_FLOAT, False, stride, vbo_vertices + 8 * 4)

            # gl.glDrawElements(gl.GL_TRIANGLES, len(index_array), gl.GL_UNSIGNED_INT, None)
        # gl.glDrawElements(gl.GL_TRIANGLES, len(index_array), gl.GL_UNSIGNED_INT, None)
        # gl.glDrawElements(gl.GL_LINES, len(index_array), gl.GL_UNSIGNED_INT, None)
        # gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(index_array))

        T = 5e-2
        # scale_bias = (math.cos(gametic*T)*10-5, math.sin(gametic*T)*10-5)
        # scale = (math.cos(gametic*T)+1)*20
        # bias = math.sin(gametic*T*math.sqrt(2))*2
        # scale_bias = (scale, bias-scale)
        # scale_bias = ((math.cos(gametic*T)+1)*10, -10)

        # scale_bias = ((math.cos(gametic*T)+1)*10, 0)
        # scale_bias = (scale_bias[0], -scale_bias[0]-1)
        # scale_bias = (0, 0)
        # scale_bias = (0, (math.cos(gametic*T))*10)
        scale_bias = (10, -11)
        # scale_bias = (20, -21)
        # scale_bias = (5, -2.5)
        # scale_bias = (5, -6)
        # scale_bias = (5, 0.5)
        # scale_bias = (7, 0.1)
        # scale_bias = (5, 0.1)
        gl.glUniform2f(gl.glGetUniformLocation(shader, b'scale_bias'), *scale_bias)
        # print(scale_bias)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glUniform1i(gl.glGetUniformLocation(shader, b'diffuseSampler'), 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, diffuse_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glUniform1i(gl.glGetUniformLocation(shader, b'normalSampler'), 1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, normal_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)

        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glUniform1i(gl.glGetUniformLocation(shader, b'colormapSampler'), 2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, colormap_texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        for picname, group in texture_groups.items():
            if picname == '-': continue
            # print(picname, group)
            load_texture(group['picture'], 255)
            # gl.glDrawElements(gl.GL_TRIANGLES, group['count']*12, gl.GL_UNSIGNED_INT, vbo_indices+group['starting index']*12)
            gl.glDrawElements(gl.GL_TRIANGLES, group['count'] * 3, gl.GL_UNSIGNED_INT,
                              vbo_indices + group['starting index'] * 4)

        for picname, group in flat_groups.items():
            if picname == '-': continue

            load_texture(group['picture'], 255)
            # gl.glDrawElements(gl.GL_TRIANGLES, group['count']*12, gl.GL_UNSIGNED_INT, vbo_indices+group['starting index']*12)
            gl.glDrawElements(gl.GL_TRIANGLES, group['count'] * 3, gl.GL_UNSIGNED_INT,
                              vbo_indices + group['starting index'] * 4)
            # gl.glDrawElements(gl.GL_TRIANGLES, group['count'], gl.GL_UNSIGNED_INT, vbo_indices+group['starting index']*4)

    sky_vertex_shader = shaders.compileShader(SKY_VERTEX_SHADER_CODE, gl.GL_VERTEX_SHADER)
    sky_fragment_shader = shaders.compileShader(SKY_FRAGMENT_SHADER_CODE, gl.GL_FRAGMENT_SHADER)
    sky_shader = shaders.compileProgram(sky_vertex_shader, sky_fragment_shader)

    JOYSTICK_SCALE = 5.0
    JOYSTICK_TURN_SCALE = 5e-1
    JOYSTICK_DEADZONE = 3e-1
    STRAFE_DISTANCE = 5.0
    FORWARD_DISTANCE = 5.0
    TURN_ANGLE = 10 * math.pi / 180
    try:
        texcoord_location = None

        gl.glPolygonMode(gl.GL_FRONT, gl.GL_LINE)
        # gl.glPolygonMode(gl.GL_FRONT, gl.GL_FILL)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            if joystick.get_name() == 'Microsoft X-Box 360 pad':
                joystick.init()
            else:
                joystick = None
        else:
            joystick = None

        # stride = 6*4+2
        # gl.glVertexAttribPointer(color_location, 3, gl.GL_FLOAT, False, stride, vbo_vertices+3*4) 

        gl.glIndexPointeri(vbo_indices)

        gametic = 0
        timebegin = time.clock()
        INTERVAL = 1 / 35
        IOD = 15
        # gl.glEnable(gl.GL_SCISSOR_TEST)
        focal_distance = None  # 10+(math.sin(gametic/50.0)+1.2)*40.0
        focal_distance = 16 * 10
        while not done:
            if STEREOSCOPIC:
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                gl.glViewport(0, 0, screenx // 2, screeny)

                gl.glLoadIdentity()
                player.move_left(IOD / 2.0)
                player.update_OpenGL()
                draw_scene()

                gl.glViewport(screenx // 2, 0, screenx // 2, screeny)

                gl.glLoadIdentity()
                player.move_left(-IOD)
                player.update_OpenGL()
                draw_scene()

                pygame.display.flip()
                player.move_left(IOD / 2.0)

            else:
                gl.glViewport(0, 0, screenx, screeny)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                gl.glLoadIdentity()
                player.update_OpenGL()
                draw_scene()
                pygame.display.flip()

            time.sleep(INTERVAL)

            gametic += 1
    finally:
        if joystick is not None:
            joystick.quit()
        vbo_indices.unbind()
        vbo_vertices.unbind()
        if texcoord_location >= 0:
            gl.glDisableVertexAttribArray(texcoord_location)
        gl.glDisableVertexAttribArray(normal_location)
        gl.glDisableVertexAttribArray(position_location)
        gl.glDeleteTextures(normal_texture)
        gl.glDeleteTextures(diffuse_texture)
        gl.glDeleteTextures(sky_texture)
        gl.glUseProgram(0)


if len(sys.argv) > 2:
    wadtype = 'doom'
    wadname = os.path.basename(sys.argv[1]).lower()
    if wadname == 'hexen.wad': wadtype = 'hexen'
    if wadname == 'strife1.wad': wadtype = 'strife'

    render_level(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, wadtype)
else:
    print('Usage: renderer.py [iwad] [map name]')