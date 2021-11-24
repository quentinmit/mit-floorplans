#!/usr/bin/env python3

import collections
import logging
import sys
import os

from more_itertools import peekable, take
import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, token, mat, transformed_bounds, transformed_points, IDENTITY, rotate_mat
from fonts import _KNOWN_SHAPES, CHAR_POINTS

logger = logging.getLogger('extractfloorplan')

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

    def debug_rect(self, bounds, text=None, rotated=False, stroke="black", fill="none", **kwargs):
        p = Path(stroke=stroke, fill=fill, **kwargs)
        if text:
            p.args['title'] = text
        p.M(bounds[0], -bounds[1])
        p.H(bounds[2])
        p.V(-bounds[3])
        p.H(bounds[0])
        p.Z()
        if rotated:
            self.stack[1].append(p)
        else:
            self.top.append(p)


    def find_north(self):
        left, top, width, height = self.top.viewBox

        north_bounds = [left+(0.7*width), top+(0.89*height), left+(0.76*width), top+height]

        if self.debug_angle:
            self.debug_rect(north_bounds, stroke="blue")

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
        if (n < 0.3).all():
            return True
        elif (n < 10).all():
            pass
            #logger.debug("possible match %s", n)
        return False

    _Shape = collections.namedtuple('_Shape', 'parent child commands points bounds'.split())

    def find_characters(self):
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Path):
                    continue
                bounds = transformed_bounds(child.bounds, matrix)
                commands = child.offset_commands()
                points = self._shape(child)
                yield self._Shape(parent, child, commands, points, bounds)

        def _shape_str(commands):
            out = []
            for cmd, args in commands:
                out.append(cmd+",".join("%g" % x for x in args))
            return "".join(out)
        path_ids = dict()
        paths_by_shape = dict()

        iterator = peekable(_iterator())
        last_was_char = False
        while iterator:
            # Try to OCR the next N shapes
            try:
                char, elements = self._ocr(iterator)
            except TypeError:
                pass
            else:
                logger.info("shape at %s: %s (%d elements)", iterator[0].bounds, char, elements)
                if char not in ('T', 'L', 'I', '-', '/') or last_was_char: # easy for walls to look like these characters
                    paths_by_shape[char] = paths_by_shape.get(char, 0) + 1
                    for i, shape in enumerate(take(elements, iterator)):
                        shape.child.args['stroke'] = 'green'
                        shape.child.args['class'] = 'vectortext'
                        shape.child.args['title'] = char + '(%s)' % (','.join("%g" % x for x in shape.points.flatten()))
                    last_was_char = True
                    continue
            # If not, catalog and move on
            last_was_char = False
            shape = next(iterator)
            shape_str = _shape_str(shape.commands)
            paths_by_shape[shape_str] = paths_by_shape.get(shape_str, 0) + 1
            if shape_str not in path_ids:
                path_ids[shape_str] = len(path_ids)
            logger.info("unknown shape at %s: %s", shape.bounds, shape_str)
            logger.debug("path id %s", path_ids[shape_str])
            logger.debug("commands %s", shape.commands)
            logger.debug("curves %r", shape.child.curves)
            #logger.debug("quantized %r", shape.child.quantize())
            shape_repr = shape_str
            if shape.points is not None:
                shape_repr = '(%s)' % (','.join("%g" % x for x in shape.points.flatten()))
            shape.child.args['title'] = "%s %s" % (path_ids[shape_str], shape_repr)

        for count, shape in sorted((v,k) for k,v in paths_by_shape.items()):
            logger.info("%d copies of %s: %s", count, path_ids.get(shape), shape)

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
        parser.parsepage(page, d)
        if parser.bogus:
            raise ValueError("missing floorplan")
        logger.info("page viewbox = %s", d.viewBox)
        logger.info("%s bounds = %s", parser.stack[1], parser.stack[1].bounds)
        parser.find_north()
        parser.find_characters()
        annot_mat = np.dot(rotate_mat(rotate/180*np.pi), mat(-1,0,0,1,width,0))
        if parser.north_angle and not parser.debug_angle:
            parser.remove_edge_content(parser.stack[1], IDENTITY)
            parser.stack[1].matrix = np.dot(rotate_mat(parser.north_angle), parser.stack[1].matrix)
            annot_mat = np.dot(rotate_mat(parser.north_angle), annot_mat)
        for a in annots:
            rect = [float(x) for x in a.Rect]
            rect = transformed_points(rect, annot_mat)[:,:2].flatten()
            parser.debug_rect(rect, text=a.Contents.to_unicode(), stroke="orange")
        print(d.asSvg())

if __name__ == '__main__':
    main()
