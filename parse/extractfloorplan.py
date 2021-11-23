#!/usr/bin/env python3

import collections
import logging
import sys
import os

from more_itertools import peekable, take
import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, token, mat, transformed_bounds, transformed_points, IDENTITY

logger = logging.getLogger('extractfloorplan')

_KNOWN_SHAPES = dict()

for shapes in [
        # 50_0 stair font
        ("U", (-27,0,-32,-1,-36,-4,-37,-8,-37,-11,-36,-15,-32,-17,-27,-19,0,-19)),
        ("P", (37,0,37,-12,35,-17,34,-18,30,-19,25,-19,21,-18,19,-17,18,-12,18,0)),
        # 50_0 room number font
        ("A", (31,-12,0,-24), (0,-15)),
        ("B", (-1,-4,-3,-5,-6,-7,-10,-7,-13,-5,-14,-4,-16,0,-16,14,15,14,15,0,14,-4,12,-5,9,-7,6,-7,3,-5,2,-4,0,0,0,14)),
        ("C", (3,2,6,5,7,8,7,14,6,17,3,20,0,21,-5,23,-12,23,-17,21,-19,20,-22,17,-24,14,-24,8,-22,5,-19,2,-17,0)),
        ("D", (31,0,31,-10,29,-15,26,-18,23,-19,19,-21,11,-21,7,-19,4,-18,1,-15,0,-10,0,0)),
        ("E", (32,0,32,-20), (0,-12), (0,-20)),
        ("F", (31,0,31,-20), (0,-12)),
        ("G", (3,2,6,5,7,8,7,14,6,17,3,20,0,21,-5,23,-12,23,-17,21,-20,20,-23,17,-24,14,-24,8,-23,5,-20,2,-17,0,-12,0,-12,8)),
        ("I", (-31,0)),
        ("J", (-24,0,-28,2,-30,3,-31,6,-31,9,-30,12,-28,14,-24,15,-21,15)),
        ("K", (-31,0), (-21,20), (-18,-13)),
        ("L", (-32,0,-32,-18)),
        ("M", (31,0,0,-12,31,-24,0,-24)),
        ("N", (31,0,0,-21,31,-21)),
        ("O", (-1,3,-4,6,-7,8,-11,9,-19,9,-23,8,-26,6,-29,3,-31,0,-31,-6,-29,-9,-26,-12,-23,-13,-19,-15,-11,-15,-7,-13,-4,-12,-1,-9,0,-6,0,0)),
        ("P", (31,0,31,-13,29,-18,28,-19,25,-21,20,-21,17,-19,16,-18,14,-13,14,0)),
        ("R", (31,0,31,-13,30,-18,28,-19,25,-21,22,-21,19,-19,18,-18,16,-13,16,0), (-16,-11)),
        ("S", (3,3,5,8,5,14,3,18,0,21,-3,21,-6,20,-7,18,-9,15,-12,6,-13,3,-15,2,-18,0,-22,0,-25,3,-26,8,-26,14,-25,18,-22,21)),
        ("T", (-32,0), (0,-21)),
        ("U", (-22,0,-27,-2,-30,-5,-31,-9,-31,-12,-30,-17,-27,-20,-22,-21,0,-21)),
        ("V", (-32,-12,0,-24)),
        ("V", (-75,-22,0,-43)),
        ("W", (-75,-13,0,-27,-75,-40,0,-53)),
        ("Y", (-36,-22,0,-43), (-39,0)),
        ("0", (-1,5,-6,8,-13,9,-17,9,-25,8,-29,5,-31,0,-31,-3,-29,-7,-25,-10,-17,-12,-13,-12,-6,-10,-1,-7,0,-3,0,0)),
        ("1", (1,-3,6,-8,-26,-8)),
        ("2", (1,0,4,-1,6,-3,7,-6,7,-12,6,-15,4,-16,1,-18,-2,-18,-5,-16,-9,-13,-24,1,-24,-19)),
        ("3", (0,-16,-12,-8,-12,-12,-14,-15,-15,-16,-20,-18,-23,-18,-27,-16,-30,-13,-32,-9,-32,-5,-30,0,-29,1,-26,3)),
        ("4", (31,0,11,15,11,-7)),
        ("5", (0,15,-13,16,-12,15,-10,10,-10,6,-12,1,-15,-2,-19,-3,-22,-3,-26,-2,-29,1,-31,6,-31,10,-29,15,-28,16,-25,18)),
        ("6", (3,1,4,6,4,9,3,13,-2,16,-9,17,-16,17,-22,16,-25,13,-27,9,-27,7,-25,3,-22,0,-18,-2,-16,-2,-12,0,-9,3,-8,7,-8,9,-9,13,-12,16,-16,17)),
        ("7", (31,-15,31,6)),
        ("8", (-2,4,-5,6,-8,6,-11,4,-12,1,-14,-5,-15,-9,-18,-12,-21,-14,-26,-14,-29,-12,-30,-11,-32,-6,-32,0,-30,4,-29,6,-26,7,-21,7,-18,6,-15,3,-14,-2,-12,-8,-11,-11,-8,-12,-5,-12,-2,-11,0,-6,0,0)),
        ("9", (-4,1,-7,4,-8,9,-8,10,-7,15,-4,18,0,19,2,19,6,18,9,15,11,10,11,9,9,4,6,1,0,0,-7,0,-14,1,-19,4,-20,9,-20,12,-19,16,-16,18)),
        ("/", (-48,27)),
        ("-", (0,-27)),
        # 50_0 footer
        ("a", (-40,0), (6,6,9,12,9,20,6,26,0,32,-8,34,-14,34,-23,32,-28,26,-31,20,-31,12,-28,6,-23,0)),
        ("c", (6,6,9,12,9,20,6,26,0,32,-8,35,-14,35,-23,32,-28,26,-31,20,-31,12,-28,6,-23,0)),
        ("d", (-60,0), (6,5,9,11,9,20,6,25,0,31,-8,34,-14,34,-23,31,-28,25,-31,20,-31,11,-28,5,-23,0)),
        ("e", (0,-34,5,-34,11,-31,14,-28,17,-23,17,-14,14,-8,8,-3,0,0,-6,0,-15,-3,-20,-8,-23,-14,-23,-23,-20,-28,-15,-34)),
        ("f", (0,6,-3,12,-12,14,-60,14), (0,-20)),
        ("g", (-46,0,-54,3,-57,6,-60,11,-60,20,-57,26), (6,6,9,11,9,20,6,26,0,31,-8,34,-14,34,-23,31,-28,26,-31,20,-31,11,-28,6,-23,0)),
        ("h", (-60,0), (9,-8,12,-14,12,-23,9,-28,0,-31,-28,-31)),
        ("i", (-3,-3,0,-6,3,-3,0,0), (-37,0)),
        ("n", (-40,0), (9,-9,12,-14,12,-23,9,-29,0,-32,-28,-32)),
        ("o", (-3,6,-9,12,-17,15,-23,15,-32,12,-37,6,-40,0,-40,-8,-37,-14,-32,-20,-23,-23,-17,-23,-9,-20,-3,-14,0,-8,0,0)),
        ("p", (-60,0), (6,-5,9,-11,9,-20,6,-25,0,-31,-8,-34,-14,-34,-23,-31,-28,-25,-31,-20,-31,-11,-28,-5,-23,0)),
        ("r", (-40,0), (8,-2,14,-8,17,-14,17,-22)),
        ("s", (6,3,9,11,9,20,6,28,0,31,-5,28,-8,23,-11,8,-14,3,-20,0,-23,0,-28,3,-31,11,-31,20,-28,28,-23,31)),
        ("t", (-49,0,-57,-3,-60,-9,-60,-14), (0,-20)),
        ("u", (-29,0,-37,-3,-40,-8,-40,-17,-37,-23,-29,-31), (-40,0)),
        ("v", (-40,-17,0,-34)),
        ("y", (-40,-17,0,-34), (-12,5,-17,11,-20,17,-20,20)),
        (".", (-3,3,-6,0,-3,-3,0,0)),

        # W20_5 room number font
        ("P", (1.92,0,1.92,-0.84,1.74,-1.2,1.56,-1.26,1.26,-1.26,1.08,-1.2,1.02,-1.08,0.9,-0.84,0.9,0)),
        ("V", (-1.86,-0.66,0,-1.38)),
        ("4", (-1.26,0.9,-1.26,-0.48), (-1.92,0)),
        #("5", (0,0.9,-0.78,0.96,-0.72,0.9,-0.6,0.6,-0.6,0.36,-0.72,0.06,-0.9,-0.12,-1.14,-0.18,-1.32,-0.18,-1.62,-0.12,-1.8,0.06,-1.86,0.36,-1.86,0.6,-1.8,0.9,-1.68,0.96,-1.5,1.08)),
        # NE83_4 room number font
        ("B", (3.78,0,3.78,-1.62,3.6,-2.16,3.42,-2.34,3.06,-2.52,2.7,-2.52,2.34,-2.34,2.16,-2.16,1.98,-1.62,1.98,0), (-0.18,-0.54,-0.36,-0.72,-0.72,-0.9,-1.26,-0.9,-1.62,-0.72,-1.8,-0.54,-1.98,0,-1.98,1.62)),
        ("Y", (-2.1,-1.74,0,-3.48), (-2.4,0)),
]:
    shapes = list(shapes)
    k = shapes.pop(0)
    l = len(shapes[0])//2
    shapes = [np.array(v, dtype='f').reshape((-1, 2)) for v in shapes]
    r = np.min(shapes[0])-np.max(shapes[0])
    shapes = [s/r for s in shapes]
    if l not in _KNOWN_SHAPES:
        _KNOWN_SHAPES[l] = list()
    _KNOWN_SHAPES[l].append((k, shapes))

for shapes in _KNOWN_SHAPES.values():
    shapes.sort(key=lambda t: len(t[1]), reverse=True)

class Floorplan2Svg(Pdf2Svg):
    bogus = False
    scale = None
    north_angle = None

    debug_angle = False

    def stack_matrix(self):
        matrix = mat(1,0,0,1,0,0)
        for element in reversed(self.stack):
            if hasattr(element, 'matrix'):
                matrix = np.dot(element.matrix, matrix)
        #logger.debug("stack matrix = %s", matrix)
        return matrix

    # def gstate(self, _matrix=None, **kwargs):
    #     if _matrix is not None:
    #         new = np.dot(_matrix, self.stack_matrix())
    #         logger.info("new gstate is at %s, height=%s", new[2], self.top.height)
    #         if new[2][1] < -0.8*self.top.height:
    #             g = Group()
    #             g.matrix = _matrix
    #             self.stack.append(g)
    #             self.last = g
    #             return
    #     super().gstate(_matrix=_matrix, **kwargs)

    @token('Tj', 't')
    def parse_text_out(self, text):
        super().parse_text_out(text)
        if isinstance(text, PdfString):
            text = text.to_unicode()
        if '" =' in text:
            self.scale = text
            logger.info("scale = %s", text)
        if "request the plan you queried" in text:
            self.bogus = True

    # def finish_path(self, *args):
    #     if self.gpath is not None:
    #         bounds = self.gpath.bounds
    #         logger.info("path %s %s bounds %s", self.gpath.args['d'], self.gpath.transform, self.gpath.bounds)
    #         maxx, maxy, _ = np.dot((self.gpath.bounds[0], self.gpath.bounds[1], 1), self.stack_matrix())
    #         logger.info("path bounds %s lowest point on path (%f,%f)", self.gpath.bounds, maxx, maxy)
    #         if False and maxy > -0.2*self.top.height:
    #             self.gpath = None
    #             return
    #     super().finish_path(*args)

    def remove_edge_content(self, parent, matrix):
        matrix = np.dot(parent.matrix, matrix)
        for child in list(parent.children):
            if isinstance(child, Group):
                self.remove_edge_content(child, matrix)
                if len(child.children) == 0:
                    parent.children.remove(child)
            else:
                bounds = transformed_bounds(child.bounds, matrix)
                if bounds[3] > -0.2*self.top.height:
                    parent.children.remove(child)

    def iterelements(self, parent=None, matrix=IDENTITY):
        if parent:
            matrix = np.dot(parent.matrix, matrix)
            children = parent.children
        else:
            parent = self.top
            children = parent.elements
        for child in children:
            if isinstance(child, Group):
                yield from self.iterelements(child, matrix)
            else:
                yield parent, child, matrix

    def find_north(self):
        left, top, width, height = self.top.viewBox

        north_bounds = [left+(0.7*width), top+(0.89*height), left+(0.76*width), top+height]

        if self.debug_angle:
            p = Path(stroke="blue", fill="none")
            p.M(north_bounds[0], -north_bounds[1])
            p.H(north_bounds[2])
            p.V(-north_bounds[3])
            p.H(north_bounds[0])
            p.Z()
            self.top.append(p)

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
                commands = child.commands
                if len(commands) == 2 and [c[0] for c in commands] == ['M', 'L']:
                    logger.info("possible north: %s bounds %s", child, bounds)
                    child.args["stroke"] = "blue"
                    line = transformed_points([c[1] for c in commands], matrix)[:,:2]
                    lines.append(line)

        lines = np.array(lines)

        vectors = lines[:,1,:]-lines[:,0,:]
        logger.debug("north vectors %s", vectors)
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        logger.info("angles = %s", angles*180/np.pi)
        if len(angles):
            self.north_angle = angles[1]

    def _shape(self, commands):
        points = []
        for cmd, args in commands:
            if cmd != 'L':
                return None
            points.append(args)
        return np.array(points)

    def _ocr(self, peekable):
        """
        Attempt to recognize a character in peekable, a peekable stream of _Shape.

        Returns: character, number of elements required or None
        """
        first = peekable.peek()
        points = first.points
        if points is None:
            return None
        div = np.min(points)-np.max(points)
        for char, char_shapes in _KNOWN_SHAPES.get(points.shape[0], []):
            for i, char_points in enumerate(char_shapes):
                try:
                    shape = peekable[i]
                except IndexError:
                    break
                if shape.parent != first.parent:
                    break
                if not self._same(char_points, shape.points/div):
                    break
            else:
                return char, len(char_shapes)

    def _same(self, a, b):
        if a.shape != b.shape:
            return False
        diff = a - b
        n = np.linalg.norm(diff, axis=1)
        if (n < 0.1).all():
            return True
        elif (n < 10).all():
            logger.debug("possible match %s", n)
        return False

    _Shape = collections.namedtuple('_Shape', 'parent child commands points bounds'.split())

    def find_characters(self):
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Path):
                    continue
                bounds = transformed_bounds(child.bounds, matrix)
                commands = child.offset_commands()
                points = []
                for cmd, args in commands:
                    if cmd != 'L':
                        points = None
                        break
                    points.append(args)
                else:
                    points = np.array(points, dtype='f').reshape((-1, 2))
                yield self._Shape(parent, child, commands, points, bounds)

        def _shape_str(commands):
            out = []
            for cmd, args in commands:
                out.append(cmd+",".join("%g" % x for x in args))
            return "".join(out)
        path_ids = dict()
        paths_by_shape = dict()

        iterator = peekable(_iterator())
        while iterator:
            # Try to OCR the next N shapes
            try:
                char, elements = self._ocr(iterator)
            except TypeError:
                pass
            else:
                logger.info("shape at %s: %s (%d elements)", shape.bounds, char, elements)
                paths_by_shape[char] = paths_by_shape.get(char, 0) + 1
                for i, shape in enumerate(take(elements, iterator)):
                    shape.child.args['stroke'] = 'green'
                    shape.child.args['title'] = char + '(%s)' % (','.join("%g" % x for x in shape.points.flatten()))
                continue
            # If not, catalog and move on
            shape = next(iterator)
            shape_str = _shape_str(shape.commands)
            paths_by_shape[shape_str] = paths_by_shape.get(shape_str, 0) + 1
            logger.info("unknown shape at %s: %s", shape.bounds, shape_str)
            if shape_str not in path_ids:
                path_ids[shape_str] = len(path_ids)
            shape_repr = shape_str
            if shape.points is not None:
                shape_repr = '(%s)' % (','.join("%g" % x for x in shape.points.flatten()))
            shape.child.args['title'] = "%s %s" % (path_ids[shape_str], shape_repr)

        for count, shape in sorted((v,k) for k,v in paths_by_shape.items()):
            logger.info("%d copies of %s: %s", count, path_ids.get(shape), shape)

    def find_north_old(self, parent, matrix):
        matrix = np.dot(parent.matrix, matrix)
        for child in parent.children:
            if isinstance(child, Group):
                self.find_north(child, matrix)
            elif isinstance(child, Path):
                bounds = transformed_bounds(child.bounds, matrix)
                if bounds[1] > -0.2*self.top.height:
                    commands = child.args['d'].split(' ')
                    if not len(commands) == 4:
                        continue
                    if not (bounds[3]-bounds[1])/(bounds[2]-bounds[0]) < 1.1:
                        continue
                    args = [[float(x) for x in c[1:].split(',')] for c in commands]
                    commands = [c[0] for c in commands]
                    if commands != ['M', 'L', 'M', 'L']:
                        continue
                    points = transformed_points(args, matrix)[:,:2]
                    logger.info("possible north: %s bounds %s", points, bounds)
                    vectors = points[::2]-points[1::2]
                    logger.info("lines = %s", vectors)
                    angles = np.arctan2(vectors[:,1], vectors[:,0])
                    logger.info("angles = %s", angles*180/np.pi)
                    child.args["stroke"] = "red"
                    # TODO: Figure out how to identify the bold rectangle indicating N
                    self.north_angle = angles[1]+(np.pi)

def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('decodegraphics').setLevel(logging.INFO)

    inpfn, = sys.argv[1:]
    outfn = 'copy.' + os.path.basename(inpfn)
    pages = PdfReader(inpfn, decompress=True).pages

    parser = Floorplan2Svg()
    parser.debug_angle = True

    sys.setrecursionlimit(sys.getrecursionlimit()*2)

    for page in pages:
        box = [float(x) for x in page.MediaBox]
        assert box[0] == box[1] == 0, "demo won't work on this PDF"
        left, bottom, width, height = box
        rotate = 0
        if '/Rotate' in page:
            rotate = int(page.Rotate)
            if rotate % 180 == 90:
                width, height = height, width
        d = draw.Drawing(width, height)
        parser.parsepage(page, d)
        if parser.bogus:
            raise ValueError("missing floorplan")
        logger.info("page viewbox = %s", d.viewBox)
        logger.info("%s bounds = %s", parser.stack[1], parser.stack[1].bounds)
        parser.find_north()
        parser.find_characters()
        if parser.north_angle and not parser.debug_angle:
            parser.remove_edge_content(parser.stack[1], IDENTITY)
            cosA = np.cos(parser.north_angle)
            sinA = np.sin(parser.north_angle)
            rotate_mat = mat(cosA,sinA, -sinA,cosA, 0,0)
            parser.stack[1].matrix = np.dot(rotate_mat, parser.stack[1].matrix)
        print(d.asSvg())

if __name__ == '__main__':
    main()
