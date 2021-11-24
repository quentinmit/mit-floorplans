#!/usr/bin/env python3

import collections
import logging
import sys
import os
from itertools import chain

from more_itertools import peekable, take, ichunked
import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, token, mat, transformed_bounds, transformed_points, IDENTITY, rotate_mat
from fonts import KNOWN_SHAPES, CHAR_POINTS

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

    def debug_rect(self, bounds, text=None, parent=None, stroke="black", fill="none", **kwargs):
        p = Path(stroke=stroke, fill=fill, **kwargs)
        if text:
            p.args['title'] = text
        p.M(bounds[0], -bounds[1])
        p.H(bounds[2])
        p.V(-bounds[3])
        p.H(bounds[0])
        p.Z()
        if parent:
            parent.append(p)
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
        div = np.max(points)-np.min(points)

        last_len = None
        possibilities = []
        for char, char_shapes in KNOWN_SHAPES:
            norms = []
            try:
                for i, char_points in enumerate(char_shapes):
                    try:
                        shape = peekable[i]
                    except IndexError:
                        break
                    if shape.parent != first.parent:
                        break
                    norms.append(self._same(char_points, (shape.points+shape.offset-first.offset)/div))
                else:
                    norm = np.concatenate(norms)
                    if (norm < 0.3).all():
                        score = np.max(norm)
                        possibilities.append((score, char, len(char_shapes)))
            except:
                logger.exception("attempting to match %s %s\nnorms %s", char, char_shapes, norms)
                raise

            if len(char_shapes) != last_len and possibilities:
                # Return the most-similar shape from the category with the highest number of strokes
                return next(iter(sorted(possibilities)))[1:]
            last_len = len(char_shapes)

        if possibilities:
            if len(possibilities) > 1:
                logger.info("multiple possibilities found for %s", [peekable[i].child for i in range(len(char_shapes))])
                for (score, char, l) in sorted(possibilities):
                    logger.info("possible %s with score %s", char, score)
            return next(iter(sorted(possibilities)))[1:]

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

    def find_characters(self):
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Path):
                    continue
                bounds = transformed_bounds(child.bounds, matrix)
                offset, commands = child.offset_commands()
                offset = self._rotate_point(*offset)
                points = self._shape(child)
                yield self._Shape(parent, child, offset, commands, points, bounds)

        def _shape_str(commands):
            out = []
            for cmd, args in commands:
                try:
                    if int(self.page.Rotate or 0) == 270:
                        if cmd == 'H':
                            cmd = 'V'
                        elif cmd == 'V':
                            cmd = 'H'
                            args = (-args[0],)
                        else:
                            args = chain.from_iterable((-y if y else 0, x) for x,y in ichunked(args, 2))
                    elif self.page.Rotate:
                        raise NotImplementedError()
                except:
                    logger.exception("failed to update %s, %s", cmd, args)
                    raise
                out.append(cmd+",".join("%g" % x for x in args))
            return "".join(out)
        path_ids = dict()
        paths_by_shape = dict()

        iterator = peekable(_iterator())
        last_was_char = False
        last_offset = None
        while iterator:
            # Try to OCR the next N shapes
            result = self._ocr(iterator)
            if result:
                char, elements = result
                logger.info("shape at %s: %s (%d elements)", iterator[0].bounds, char, elements)
                if char not in ('T', 'L', 'I', '-', '/', 'O', 'V', 'U') or last_was_char: # easy for walls to look like these characters
                    paths_by_shape[char] = paths_by_shape.get(char, 0) + 1
                    for i, shape in enumerate(take(elements, iterator)):
                        if i == 0:
                            last_offset = shape.child.offset
                        shape.child.args['stroke'] = 'green'
                        shape.child.args['class'] = 'vectortext'
                        shape.child.args['title'] = char + ' %s (%s)' % (_shape_str(shape.commands), ','.join("%g" % x for x in shape.points.flatten()))
                    last_was_char = True
                    continue
            # If not, catalog and move on
            last_was_char = False
            shape = next(iterator)
            shape_str = _shape_str(shape.commands)
            paths_by_shape[shape_str] = paths_by_shape.get(shape_str, 0) + 1
            if shape_str not in path_ids:
                path_ids[shape_str] = len(path_ids)
            logger.info("unknown shape %s at %s: %s", path_ids[shape_str], shape.bounds, shape_str)
            if path_ids[shape_str] in (236,):
                logger.debug("commands %s", shape.child.commands)
                logger.debug("offset commands %s", shape.commands)
                logger.debug("curves %r", shape.child.curves)
                logger.debug("offset %s", shape.child.offset)
                logger.debug("last offset %s", last_offset)
                logger.debug("rotated offset %s", shape.offset)
                #logger.debug("quantized %r", shape.child.quantize())
                logger.debug("points %r", shape.points)
            shape_repr = shape_str
            if last_offset:
                shape_repr += ' or %s' % _shape_str(shape.child.offset_commands(last_offset)[1])
            if shape.points is not None:
                shape_repr += ' (%s)' % (','.join("%g" % x for x in shape.points.flatten()))
            shape.child.args['title'] = "%s %s" % (path_ids[shape_str], shape_repr)
            last_offset = shape.child.offset

        for count, shape in sorted((v,k) for k,v in paths_by_shape.items()):
            logger.info("%d copies of %s: %s", count, path_ids.get(shape), shape)

def main():
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('decodegraphics').setLevel(logging.INFO)

    logging.debug("known shapes = %s", KNOWN_SHAPES)

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
        if annots:
            annotg = Group(class_="annotations")
            d.append(annotg)
        for a in annots:
            rect = [float(x) for x in a.Rect]
            rect = transformed_points(rect, annot_mat)[:,:2].flatten()
            parser.debug_rect(rect, text=a.Contents.to_unicode(), parent=annotg, stroke="orange")
        print(d.asSvg())

if __name__ == '__main__':
    main()
