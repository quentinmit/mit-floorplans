#!/usr/bin/env python3

import argparse
import collections
from itertools import chain, count
import json
import logging
import os
import re
import sys

from more_itertools import peekable, take, ichunked
import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, Text, token, mat, transformed_bounds, transformed_points, IDENTITY, rotate_mat
from fonts import KNOWN_SHAPES, CHAR_POINTS

logger = logging.getLogger('extractfloorplan')

_GROUP_STYLES = {
    'A-WALL': # Regular walls
    "stroke: purple",
    'A-WALL-PHTM': # Phantom walls (cages?)
    "stroke: gray",
    'A-WALL-PRHT': # Partial height walls
    "stroke: gray",
    'A-WALL-CORE': # Perimeter walls and foundations
    "",
    'A-GLAZ': # Windows
    "stroke: orange",
    'A-DOOR': # Interior and exterior doors
    "stroke: blue",
    'A-MISC': # Stairs (etc.)
    "",
    'A-AREA-FICM$TXT': # Text inside floorplan
    "",
}

class Floorplan2Svg(Pdf2Svg):
    logger = logger

    bogus = False
    scale = None
    north_angle = None

    debug_angle = False

    def __init__(self):
        super().__init__()
        self.seen_ocg_names = set()

    def start_optional_content_group(self, class_):
        self.seen_ocg_names.add(class_)

    def stack_matrix(self):
        matrix = mat(1,0,0,1,0,0)
        for element in reversed(self.stack):
            if hasattr(element, 'matrix'):
                matrix = np.dot(element.matrix, matrix)
        #logger.debug("stack matrix = %s", matrix)
        return matrix

    def do_text(self, text):
        super().do_text(text)
        if '" =' in text:
            self.scale = text
            logger.info("text scale = %s", text)
        if "request the plan you queried" in text:
            self.bogus = True

    def remove_edge_content(self, parent, matrix):
        matrix = np.dot(parent.matrix, matrix)
        for child in list(parent.children):
            if isinstance(child, Group):
                self.remove_edge_content(child, matrix)
                if len(child.children) == 0:
                    parent.children.remove(child)
            else:
                bounds = transformed_bounds(child.bounds, matrix)
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                if isinstance(child, Path) and width == 0 and height == 0:
                    parent.children.remove(child)
                    logger.warning("Removing zero-size path %s", child)
                elif bounds[3] > -0.2*self.top.height:
                    parent.children.remove(child)
                elif bounds[0] > self.top.viewBox[0]+(0.9*self.top.width):
                    parent.children.remove(child)

    def iterelements(self, parent=None, matrix=IDENTITY, group_filter=lambda parent, child, matrix: True):
        if parent:
            matrix = np.dot(parent.matrix, matrix)
            children = parent.children
        else:
            parent = self.top
            children = parent.elements
        for child in list(children): # snapshot in case list is mutated mid-iteration
            yield parent, child, matrix
            if isinstance(child, Group) and group_filter(parent, child, matrix):
                yield from self.iterelements(child, matrix)

    def debug_rect(self, bounds, matrix=None, text=None, parent=None, stroke="black", fill="none", **kwargs):
        p = Path(stroke=stroke, fill=fill, **kwargs)
        if text:
            p.args['title'] = text
        points = [
            [bounds[0], bounds[1]],
            [bounds[2], bounds[1]],
            [bounds[2], bounds[3]],
            [bounds[0], bounds[3]],
            ]
        if matrix is not None:
            points = transformed_points(points,matrix)[:,:2]
        p.M(points[0][0], -points[0][1])
        for point in points[1:]:
            p.L(point[0], -point[1])
        p.Z()
        if parent:
            parent.append(p)
        else:
            self.top.append(p)


    def find_north(self):
        left, top, width, height = self.top.viewBox

        north_bounds = [left+(0.7*width), top+(0.89*height), left+(0.76*width), top+height]

        if self.debug_angle:
            self.debug_rect(transformed_bounds(north_bounds, np.linalg.inv(self.pdf2svg.matrix)), parent=self.pdf2svg, stroke="blue")

        logger.info("North expected within %s", north_bounds)

        lines = []
        for parent, child, matrix in self.iterelements():
            if not isinstance(child, Path):
                continue
            try:
                bounds = transformed_bounds(child.bounds, matrix)
            except:
                logger.exception("bounds = %s, matrix = %s", child.bounds, matrix)
                raise
            if bounds[0] > north_bounds[0] and bounds[1] > north_bounds[1] and bounds[2] < north_bounds[2] and bounds[3] < north_bounds[3]:
                if len(child.commands) % 2 == 0 and all([c[0] for c in commands] == ['M', 'L'] for commands in ichunked(child.commands, 2)):
                    for commands in ichunked(child.commands, 2):
                        logger.info("possible north: %s bounds %s", child, bounds)
                        child.args["stroke"] = "blue"
                        line = transformed_points([c[1] for c in commands], matrix)[:,:2]
                        lines.append(line)

        if not lines:
            return
        lines = np.array(lines)

        vectors = lines[:,1,:]-lines[:,0,:]
        logger.debug("north vectors %s", vectors)
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        logger.info("angles = %s", angles*180/np.pi)
        if len(angles):
            self.north_angle = angles[1]
        # TODO: Figure out how to identify the bold rectangle pointing to north

    def _shape(self, path):
        a = path.quantize(CHAR_POINTS)
        if (angle := int(self.page.Rotate or 0)) != 0:
            a = transformed_points(a, rotate_mat(-angle/180*np.pi))[:,:2]
        return a

    def _ocr(self, peekable, last_was_char):
        """
        Attempt to recognize a character in peekable, a peekable stream of _Shape.

        Returns: character, number of elements required or None
        """
        ocr_logger = logger.getChild("ocr")
        first = peekable.peek()
        points = first.points
        if points is None:
            return None
        div = np.max(points)-np.min(points)

        last_len = None
        possibilities = []
        debug = (first.bounds == (611.1599999999997, -399.17999999999984, 612.4199999999997, -397.3199999999998))
        if debug:
            ocr_logger.debug("Attempting to match character at %s", first.bounds)
            for i in range(3):
                ocr_logger.debug("peekable[%d] = %s", i, peekable[i])
                ocr_logger.debug("curves = %s", peekable[i].child.curves)
        for char in KNOWN_SHAPES:
            if len(char.shapes) != last_len and possibilities:
                if debug:
                    ocr_logger.debug("finished shapes of len %s, now %s, got possibilities %s", last_len, len(char.shapes), possibilities)
                # Return the most-similar shape from the category with the highest number of strokes
                return next(iter(sorted(possibilities)))[1:]
            last_len = len(char.shapes)

            if char.char in ('(', ')', 'J') and len(first.commands) <= 2:
                # These shapes are too similar to everything else
                continue
            norms = []
            try:
                for i, char_points in enumerate(char.shapes):
                    try:
                        shape = peekable[i]
                    except IndexError:
                        break
                    #if shape.parent != first.parent:
                    #    break
                    norms.append(self._same(char_points, (shape.points+shape.offset-first.offset)/div))
                else:
                    norm = np.concatenate(norms)
                    score = np.max(norm**2)
                    if debug:
                        ocr_logger.debug("considered %s, score %s", char.char, score)
                    if score < 0.025:
                        #score = np.max(norm)
                        # Add 1 to every score that isn't a perfect size match
                        same_size = abs(char.divisor/div - 1) < 0.01
                        if same_size:
                            ocr_logger.info("perfect match %s divisors %s %s", char.char, char.divisor, div)
                        if last_was_char or same_size or char.char not in ('T', 'L', 'I', 'l', '-', '/', 'O', 'V', 'U'):
                            score += (not same_size)*1
                            possibilities.append((score, char.char, len(char.shapes)))
            except:
                ocr_logger.exception("attempting to match %s\nnorms %s", char, norms)
                raise

        if possibilities:
            if len(possibilities) > 1:
                ocr_logger.info("multiple possibilities found for %s", [peekable[i].child for i in range(len(char.shapes))])
                for (score, char, l) in sorted(possibilities):
                    ocr_logger.info("possible %s with score %s", char, score)
            result = next(iter(sorted(possibilities)))
            first.child.args['data-ocr'] = "score %s num possibilities %d" % (result[0], len(possibilities))
            return result[1:]

    def _same(self, a, b):
        if a.shape != b.shape:
            raise ValueError("unexpected shapes: %s and %s" % (a.shape, b.shape))
        diff = a - b
        return np.linalg.norm(diff, axis=1)

    _Shape = collections.namedtuple('_Shape', 'parent child offset commands points bounds'.split())

    def _rotate_point(self, x, y):
        rotate = int(self.page.Rotate or 0)
        if rotate == 90:
            return -x, y
        elif rotate == 180:
            return -x, -y
        elif rotate == 270:
            return (-y if y else 0, x)
        else:
            return x, y

    def _shape_str(self, commands):
        out = []
        for cmd, args in commands:
            try:
                rotate = int(self.page.Rotate or 0)
                if rotate == 270:
                    if cmd == 'H':
                        cmd = 'V'
                    elif cmd == 'V':
                        cmd = 'H'
                        args = (-args[0],)
                    else:
                        args = chain.from_iterable((-y if y else 0, x) for x,y in ichunked(args, 2))
                elif rotate:
                    raise NotImplementedError()
            except:
                logger.exception("failed to update %s, %s", cmd, args)
                raise
            out.append(cmd+",".join("%g" % x for x in args))
        return "".join(out)

    def _mark_character(self, char, shapes, last_offset):
        parent = shapes[0].parent
        g = Group(class_="vectorchar", stroke='green', title=char)
        g.vectorchar = char
        parent.children.insert(parent.children.index(shapes[0].child), g)
        for shape in shapes:
            shape.parent.children.remove(shape.child)
            g.children.append(shape.child)
            shape.child.args['class'] = 'vectorchar'
            shape.child.args['title'] = char + ' %s or %s (%s)' % (self._shape_str(shape.commands), self._shape_str(shape.child.offset_commands(last_offset)[1]), ','.join("%g" % x for x in shape.points.flatten()))

    def find_characters(self):
        ocr_logger = logger.getChild('ocr')

        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Path):
                    continue
                bounds = transformed_bounds(child.bounds, matrix)
                offset, commands = child.offset_commands()
                offset = self._rotate_point(*offset)
                points = self._shape(child)
                yield self._Shape(parent, child, offset, commands, points, bounds)
        path_ids = dict()
        paths_by_shape = dict()

        iterator = peekable(_iterator())
        last_was_char = False
        last_offset = None
        while iterator:
            # Try to OCR the next N shapes
            result = self._ocr(iterator, last_was_char)
            if result:
                char, elements = result
                ocr_logger.info("shape at %s: %s (%d elements)", iterator[0].bounds, char, elements)
                paths_by_shape[char] = paths_by_shape.get(char, 0) + 1
                shapes = list(take(elements, iterator))
                self._mark_character(char, shapes, last_offset)
                last_offset = shapes[0].child.offset
                last_was_char = True
                continue
            # If not, catalog and move on
            last_was_char = False
            shape = next(iterator)
            shape_str = self._shape_str(shape.commands)
            paths_by_shape[shape_str] = paths_by_shape.get(shape_str, 0) + 1
            if shape_str not in path_ids:
                path_ids[shape_str] = len(path_ids)
            ocr_logger.info("unknown shape %s at %s: %s", path_ids[shape_str], shape.bounds, shape_str)
            if path_ids[shape_str] in (236,):
                ocr_logger.debug("commands %s", shape.child.commands)
                ocr_logger.debug("offset commands %s", shape.commands)
                ocr_logger.debug("curves %r", shape.child.curves)
                ocr_logger.debug("offset %s", shape.child.offset)
                ocr_logger.debug("last offset %s", last_offset)
                ocr_logger.debug("rotated offset %s", shape.offset)
                #ocr_logger.debug("quantized %r", shape.child.quantize())
                ocr_logger.debug("points %r", shape.points)
            shape_repr = shape_str
            if last_offset:
                shape_repr += ' or %s' % self._shape_str(shape.child.offset_commands(last_offset)[1])
            if shape.points is not None:
                shape_repr += ' (%s)' % (','.join("%g" % x for x in shape.points.flatten()))
            shape.child.args['title'] = "%s %s" % (path_ids[shape_str], shape_repr)
            last_offset = shape.child.offset

        for count, shape in sorted((v,k) for k,v in paths_by_shape.items()):
            ocr_logger.info("%d copies of %s: %s", count, path_ids.get(shape), shape)

    _SCALE_RE = re.compile('\s*=\s*'.join([r"""((?:\d+'-?)?\d+(?:/\d+)?")"""]*2))

    def _mark_text(self, text, chars):
        # FIXME: Some of these might be real
        if len(chars) <= 1:
            return
        if self._SCALE_RE.match(text.replace(" ", "")) and not self.scale:
            self.scale = text.replace(" ", "")
            logger.info("ocr scale = %s", self.scale)
        parent = chars[0].parent
        g = Group(class_="vectortext", stroke='blue', title=text)
        g.vectortext = text
        parent.children.insert(parent.children.index(chars[0].child), g)
        for char in chars:
            char.parent.children.remove(char.child)
            g.children.append(char.child)
            del char.child.args['stroke']

    _Element = collections.namedtuple('_Element', 'parent child bounds'.split())

    def find_text(self):
        logger = self.logger.getChild('find_text')
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Group) or 'vectorchar' not in child.args.get('class', ''):
                    continue
                bounds = transformed_bounds(child.bounds, matrix)
                yield self._Element(parent, child, bounds)

        iterator = peekable(_iterator())
        while iterator:
            first = next(iterator)
            chars = [first]
            line_bounds = char_bounds = first.bounds
            char_height = char_bounds[3]-char_bounds[1]
            char_width = max(char_bounds[2]-char_bounds[0], .8*char_height)
            char_height = max(char_height, char_width)
            line_width = char_width
            text = first.child.vectorchar
            if text in ('-', 'l', '/'):
                # These characters are not allowed to start a new text block.
                continue
            while iterator:
                char = iterator.peek()
                # _Char.bounds is in SVG space, so positive y is down

                # Compute the distance from the right edge of previous
                # character to the left edge of this character, as a fraction
                # of the previous character's width.
                # Compute the distance from the top edge of previous character
                # to the top edge of this character, as a fraction of the
                # previous character's height.
                char_width = max(char_width, char_bounds[2]-char_bounds[0])
                char_height = max(char_height, char_bounds[3]-char_bounds[1])
                line_width = max(line_width, line_bounds[2]-line_bounds[0])
                char_distance_x = (char.bounds[0] - char_bounds[2]) / char_width
                char_distance_y = (char.bounds[1] - char_bounds[1]) / char_height
                # Ditto but with x/y swapped
                line_distance_x = (char.bounds[0] - line_bounds[0]) / line_width
                line_distance_y = (char.bounds[1] - line_bounds[3]) / char_height #(line_bounds[3]-line_bounds[1])

                if 0 < char_distance_x < 2 and -.75 < char_distance_y < .75:
                    if char_distance_x > 1:
                        text += ' '
                    line_bounds = (min(line_bounds[0], char.bounds[0]), min(line_bounds[1], char.bounds[1]), max(line_bounds[2], char.bounds[2]), max(line_bounds[3], char.bounds[3]))
                elif -1 < line_distance_x < .5 and 0 < line_distance_y < 1:
                    text += '\n'
                    line_bounds = char.bounds
                else:
                    logger.debug('after text %s considered %s, char dist = (%g,%g) line dist = (%g,%g)', text, char, char_distance_x, char_distance_y, line_distance_x, line_distance_y)
                    logger.debug('previous bounds were char %s line %s', char_bounds, line_bounds)
                    break

                # Fix characters that are indistinguishable from each other.
                if char.child.vectorchar == 'I' and text[-1].islower():
                    char.child.vectorchar = 'l'
                elif char.child.vectorchar in ('O', 'o', '0'):
                    logger.info('potential misrecognized character %s after "%s": ratio %g', char.child.vectorchar, text, char.bounds.size / char_bounds.size)
                    if char.bounds.size / char_bounds.size < .25:
                        # Misrecognized .
                        char.child.vectorchar = '.'
                chars.append(next(iterator))
                text += char.child.vectorchar
                char_bounds = char.bounds

            logger.debug('vector text "%s" found at %s: %s', text, first.bounds[:2], chars)
            self._mark_text(text, chars)

    def reify_text(self, fontSize=100):
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Group) or 'vectortext' not in child.args.get('class', ''):
                    continue
                bounds = child.projected_bounds(matrix)
                yield self._Element(parent, child, bounds)

        g = Group(
            class_="text",
#            text_align='center',
#            text_anchor='middle',
        )
        self.top.append(g)

        for element in _iterator():
            center = np.mean(np.array(element.bounds).reshape((-1,2)), axis=0)
            g.append(Text(
                text=element.child.vectortext,
                fontSize=fontSize,
                x=center[0],
                y=-center[1],
                center=True,
                valign='middle',
            ))

    def connect_walls(self):
        path_segments_by_end_point = {}
        args_by_parent = {}
        for parent, child, matrix in self.iterelements():
            args_by_parent[id(child)] = dict(args_by_parent.get(id(parent), {}).items())
            for k,v in child.args.items():
                if k == 'd': continue
                args_by_parent[id(child)][k] = args_by_parent[id(child)].get(k, ()) + (v,)
            if isinstance(child, Path):
                points = child._points
                first = points[0]
                last = points[-1]
                key = (last, frozenset(args_by_parent[id(parent)].items()))
                if key in path_segments_by_end_point:
                    logger.warning("Found multiple paths ending at %s", key)
                else:
                    path_segments_by_end_point[key] = (parent, child)

        for parent, child, matrix in self.iterelements():
            if not isinstance(child, Path):
                continue
            first = child._points[0]
            key = (first, frozenset(args_by_parent[id(parent)].items()))
            previous = path_segments_by_end_point.get(key)
            if not previous:
                continue
            if child == previous[1]:
                continue
            logger.info("appended %s to %s", child, previous)
            previous[1].args['d'] += ' ' + child.args['d']
            parent.children.remove(child)
            path_segments_by_end_point[(child._points[-1], frozenset(args_by_parent[id(parent)].items()))] = previous

    def connect_walls_2(self):
        path_segments_by_args = {}
        args_by_parent = {}
        for parent, child, matrix in self.iterelements():
            args_by_parent[id(child)] = dict(args_by_parent.get(id(parent), {}).items())
            for k,v in child.args.items():
                if k == 'd': continue
                args_by_parent[id(child)][k] = args_by_parent[id(child)].get(k, ()) + (v,)
            if isinstance(child, Path):
                key = frozenset(args_by_parent[id(child)].items())
                if key not in path_segments_by_args:
                    path_segments_by_args[key] = []
                path_segments_by_args[key].append((parent, child))
        logger.info("path segments by args: %s", path_segments_by_args)

        for args_key, paths in path_segments_by_args.items():
            groups = []
            groups_by_point = {}
            for path in paths:
                logger.debug("examining %s", path)
                group = None
                for point in path[1]._points:
                    if point in groups_by_point:
                        if group:
                            oldgroup = groups_by_point[point]
                            if oldgroup == group:
                                continue
                            logger.debug("merging two groups")
                            group.extend(oldgroup)
                            for k, v in groups_by_point.items():
                                if v == oldgroup:
                                    groups_by_point[k] = group
                            groups.remove(oldgroup)
                        else:
                            group = groups_by_point[point]
                            logger.debug("point %s found in group %s", point, group)
                            group.append(path)
                if not group:
                    logger.debug("no points found, creating new group")
                    group = [path]
                    groups.append(group)
                for point in path[1]._points:
                    groups_by_point[point] = group
            logger.info("for args %s, groups:", args_key)
            for group in groups:
                if len(group) <= 1:
                    continue
                logger.info("- %r", group)
                self._stitch(group)
                for parent, child in group[1:]:
                    parent.children.remove(child)

    def _stitch(self, paths):
        """
        paths: list of (parent, child) tuples
        """

        children = set(p[1] for p in paths)

        debug = any((545.0400000000001,752.5200000000004) in p._points for p in children)

        paths_by_point = collections.defaultdict(set)
        for _, child in paths:
            points = child._points
            first, last = points[0], points[-1]
            # Don't know which direction
            paths_by_point[first].add((False, child))
            paths_by_point[last].add((True, child))

        if debug:
            logger.debug("paths = %s", children)
            logger.debug("paths_by_point = %r", paths_by_point)
        out = paths[0][1]
        children.remove(out)
        logger.info("starting with path %s", out)
        last_angle = out.final_angle
        first_point, last_point = out._points[0], out._points[-1]
        first_angle = out.initial_angle + np.pi
        # Stitch forward
        while last_point in paths_by_point:
            angles = {
                (
                    (child.final_angle-np.pi if direction else child.initial_angle),
                    direction,
                    child,
                )
                for direction, child in paths_by_point[last_point]
                if child in children
            }
            if not angles:
                break
            angles = {
                (
                    (-(angle-(last_angle+np.pi))+(2*np.pi)) % (2*np.pi),
                    angle,
                    len(child.args['d']),
                    direction,
                    i,
                    child,
                ) for i, (angle, direction, child) in enumerate(angles)
            }
            if len(angles) > 1:
                logger.info("found multiple connections at %s: %s", last_point, angles)
            _, angle, _, flipped, _, child = next(iter(sorted(angles)))
            logger.info("appended %s with initial angle %s", child, angle)
            children.remove(child)
            # Strip the M command
            if flipped:
                last_point = child._points[0]
                last_angle = child.initial_angle+np.pi
                out.args['d'] += ' ' + child.reversed_d.split(' ', 1)[1]
            else:
                last_point = child._points[-1]
                last_angle = child.final_angle
                out.args['d'] += ' ' + child.args['d'].split(' ', 1)[1]
        # Stitch backward
        if children:
            while first_point in paths_by_point:
                angles = {
                    (
                        (child.initial_angle-np.pi if direction else child.initial_angle),
                        direction,
                        child,
                    )
                    for direction, child in paths_by_point[first_point]
                    if child in children
                }
                if not angles:
                    break
                angles = {
                    (
                        ((angle-(first_angle+np.pi))+(2*np.pi)) % (2*np.pi),
                        angle,
                        len(child.args['d']),
                        direction,
                        i,
                        child,
                    ) for i, (angle, direction, child) in enumerate(angles)
                }
                if len(angles) > 1:
                    logger.info("found multiple connections at %s: %s", last_point, angles)
                _, angle, _, flipped, _, child = next(iter(sorted(angles)))
                logger.info("prepending %s with initial angle %s", child, angle)
                children.remove(child)
                first_angle = angle
                # Strip the M command
                if not flipped:
                    first_point = child._points[-1]
                    out.args['d'] = child.reversed_d + ' ' + out.args['d'].split(' ', 1)[1]
                else:
                    first_point = child._points[0]
                    out.args['d'] = child.args['d'] + ' ' + out.args['d'].split(' ', 1)[1]
        if children:
            logger.warning("disconnected children %s", children)
            for child in children:
                out.args['d'] += ' ' + child.args['d']

    _SCALE_PART_RE = re.compile(r"""(?:(?P<feet>\d+)'-?)?(?P<numerator>\d+)(?:/(?P<denominator>\d+))?"$""")
    def _parse_scale(self, text):
        text = text.strip()
        m = self._SCALE_PART_RE.match(text)
        if not m:
            logging.error("failed to parse scale %s", text)
            raise ValueError("invalid scale")
        inches = int(m.group("numerator"))
        if denominator := m.group("denominator"):
            inches /= int(denominator)
        if feet := m.group("feet"):
            inches += 12 * int(feet)
        return inches

    def apply_scale(self):
        if not self.scale:
            return
        # scale should be a string of the form 1/32" = 1'0"
        fake, real = self._SCALE_RE.match(self.scale).groups()
        # PDF default space is in units of 1/72"
        # We want to end up in a space with units of m
        fake = self._parse_scale(fake) * 72
        real = self._parse_scale(real) * 2.54 / 100
        scale = real/fake
        logger.info("applying scale of %g (%g pt -> %g m)", scale, fake, real)

        logger.info("pre-scale bounds %s viewbox %s", self.pdf2svg.bounds, self.top.viewBox)

        self.pdf2svg.apply_matrix(mat(scale, 0, 0, scale, 0, 0))

        self.fix_viewbox()

    def recenter(self):
        bounds = self.pdf2svg.projected_bounds()
        center = np.mean(np.array(bounds).reshape((-1, 2)), axis=0)
        self.apply_offset(-center[0], -center[1])

    def apply_offset(self, easting, northing):
        self.pdf2svg.apply_matrix(mat(1, 0, 0, 1, easting, northing))
        self.fix_viewbox()

    def fix_viewbox(self):
        bounds = self.pdf2svg.projected_bounds()
        self.top.viewBox = (bounds[0], bounds[1], width := bounds[2]-bounds[0], height := bounds[3]-bounds[1])
        logger.info("new bounds %s viewbox %s", bounds, self.top.viewBox)
        self.top.width = int(width)
        self.top.height = int(height)

        # TODO: Apply EPSG:26986
        # NAD83 / Massachusetts Mainland
        # https://spatialreference.org/ref/epsg/nad83-massachusetts-mainland/

    _ROOM_RE = re.compile(r"^(\d\d\d[0-9A-Z]*)($|\n)")

    def find_rooms(self, rooms, fontSize):
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Text):
                    continue
                bounds = child.projected_bounds(matrix)
                yield self._Element(parent, child, bounds)

        g = Group(class_="rooms")
        self.top.append(g)

        seen = dict()
        for element in _iterator():
            center = element.child.args["x"], element.child.args["y"]

            m = self._ROOM_RE.match(element.child.text)
            if m:
                roomNumber = m.group(1)
                seen[roomNumber] = {
                    "easting_offset": center[0],
                    "northing_offset": -center[1],
                }
                element.parent.children.remove(element.child)
                room = rooms.get(roomNumber, {})
                classes = ['room']
                for c in (room.get('major_use_desc'), room.get('use_desc'), room.get('organization_name')):
                    if c:
                        classes.append(''.join(filter(lambda x: x.isalnum(), c)))
                if not room:
                    classes.append('unknown')
                g.append(Text(roomNumber, fontSize=fontSize, class_=' '.join(classes), x=center[0], y=-center[1], center=True, valign="middle"))
        return seen

class Floorplans:
    def __init__(self, args):
        self.data = None
        self.args = args
        self.center = (0, 0)

    def load_data(self, fname):
        self.data = json.load(open(fname, 'r'))
        lobby10 = [b for b in self.data.get('buildings', []) if b['building_number'] == '10'][0]
        # Facilities data is in feet; convert to meters when needed
        self.center = lobby10['easting_x_spcs'], lobby10['northing_y_spcs']

    _FILENAME_RE = re.compile(r"^([^_]+)_([^_.]+).pdf$")

    def parse_filename(self, inpfn):
        m = self._FILENAME_RE.match(os.path.basename(inpfn))
        if m:
            return m.group(1), m.group(2)

    def fac_rooms(self, building, floor):
        return {
            room["room"]: room
            for room in self.data.get("fac_rooms", [])
            if room["building_key"] == building and room["floor"] == floor
        }

    def process_pdf(self, inpfn):
        building, floor = self.parse_filename(inpfn)

        outfn = os.path.splitext(os.path.basename(inpfn))[0] + '.svg'

        logger.info("processing %s to %s", inpfn, outfn)

        pages = PdfReader(inpfn, decompress=True).pages

        if len(pages) != 1:
            logging.error("got %d pages in %s, want 1", len(pages), inpfn)
            raise ValueError("unexpected number of pages")

        page = pages[0]

        box = [float(x) for x in page.MediaBox]
        assert box[0] == box[1] == 0, "MediaBox doesn't start at 0,0"
        annots = page.Annots or []
        for a in annots:
            logger.info("annot at %s: %s", a.Rect, a.Contents.to_unicode())

        left, bottom, width, height = box
        rotate = 0
        if '/Rotate' in page:
            rotate = int(page.Rotate)
            if rotate % 180 == 90:
                width, height = height, width
        d = draw.Drawing(width, height)
        parser = Floorplan2Svg()
        parser.debug_angle = self.args.debug_angle
        parser.parsepage(page, d)
        if parser.bogus:
            raise ValueError("missing floorplan")
        logger.info("page viewbox = %s", d.viewBox)
        logger.info("%s bounds = %s", parser.stack[1], parser.stack[1].bounds)

        # Find North before connecting paths
        parser.find_north()

        if self.args.connect_walls:
            parser.connect_walls()
        if self.args.connect_walls_2:
            parser.connect_walls_2()

        parser.find_characters()
        if not self.args.disable_find_text:
            parser.find_text()
        if parser.north_angle and not parser.debug_angle:
            parser.remove_edge_content(parser.stack[1], IDENTITY)
            parser.pdf2svg.apply_matrix(rotate_mat(-parser.north_angle))
            parser.fix_viewbox()
            #parser.stack[1].matrix = np.dot(rotate_mat(parser.north_angle), parser.stack[1].matrix)
        parser.apply_scale()
        parser.recenter()
        for b in self.data.get("buildings", []):
            if b["building_number"] == building:
                # Facilities data is in feet
                easting = (b.get("easting_x_spcs") - self.center[0])*0.3048
                northing = (b.get("northing_y_spcs") - self.center[1])*0.3048
                if northing and easting:
                    logger.info("applying easting %s northing %s", easting, northing)
                    parser.apply_offset(easting, -northing)
                else:
                    logging.error("Building %s does not have an offset: %s", building, b)
                break
        else:
            logging.error("Unknown building %s", building)
        fontSize = 6
        if parser.scale:
            fontSize = 1 # Make characters 1m by default
        if not self.args.disable_reify_text:
            parser.reify_text(fontSize)
            rooms = self.fac_rooms(building, floor)
            seen = parser.find_rooms(rooms, fontSize)
            logger.info("Found rooms: %s", seen)
            logger.info("Missing rooms: %s", set(rooms)-set(seen))
        if annots and self.args.debug_annotations:
            annotg = Group(class_="annotations")
            d.append(annotg)
            for a in annots:
                rect = [float(x) for x in a.Rect]
                parser.debug_rect(rect, matrix=parser.pdf2svg.matrix, text=a.Contents.to_unicode(), parent=annotg, stroke="orange")
        d.saveSvg(outfn)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MIT floorplans to georeferenced SVGs.')
    parser.add_argument('pdfs', metavar='PDF', nargs='+', help='PDF file to process')
    parser.add_argument('--debug-angle', action='store_true',
                        help='debug north rotation')
    parser.add_argument('--debug-annotations', action='store_true')
    parser.add_argument('--disable-find-text', action='store_true')
    parser.add_argument('--disable-reify-text', action='store_true')
    parser.add_argument('--connect-walls', action='store_true')
    parser.add_argument('--connect-walls-2', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('decodegraphics').setLevel(logging.INFO)
    logging.getLogger('extractfloorplan.ocr').setLevel(logging.WARNING)
    logging.getLogger('extractfloorplan.find_text').setLevel(logging.WARNING)

    logging.debug("known shapes = %s", KNOWN_SHAPES)

    sys.setrecursionlimit(sys.getrecursionlimit()*2)

    f = Floorplans(args)
    if args.pdfs:
        f.load_data(os.path.join(os.path.dirname(args.pdfs[0]), "data.json"))
    for inpfn in args.pdfs:
        f.process_pdf(inpfn)

if __name__ == '__main__':
    main()
