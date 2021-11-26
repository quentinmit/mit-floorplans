#!/usr/bin/env python3

import argparse
import collections
import logging
import sys
import os
from itertools import chain, count

from more_itertools import peekable, take, ichunked
import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, Text, token, mat, transformed_bounds, transformed_points, IDENTITY, rotate_mat
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
        for child in list(children): # snapshot in case list is mutated mid-iteration
            yield parent, child, matrix
            if isinstance(child, Group):
                yield from self.iterelements(child, matrix)

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
        debug = False
        for char, char_shapes in KNOWN_SHAPES:
            if len(char_shapes) != last_len and possibilities:
                if debug:
                    logger.debug("finished shapes of len %s, now %s, got possibilities %s", last_len, len(char_shapes), possibilities)
                # Return the most-similar shape from the category with the highest number of strokes
                return next(iter(sorted(possibilities)))[1:]
            last_len = len(char_shapes)

            if char in ('(', ')', 'J') and len(first.commands) <= 2:
                # These shapes are too similar to everything else
                continue
            norms = []
            try:
                for i, char_points in enumerate(char_shapes):
                    try:
                        shape = peekable[i]
                    except IndexError:
                        break
                    #if shape.parent != first.parent:
                    #    break
                    norms.append(self._same(char_points, (shape.points+shape.offset-first.offset)/div))
                else:
                    norm = np.concatenate(norms)
                    score = np.mean(norm**2)
                    if debug:
                        logger.debug("considered %s, score %s", char, score)
                    if score < 0.03:
                        #score = np.max(norm)
                        possibilities.append((score, char, len(char_shapes)))
            except:
                logger.exception("attempting to match %s %s\nnorms %s", char, char_shapes, norms)
                raise

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

    def _shape_str(self, commands):
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
            result = self._ocr(iterator)
            if result:
                char, elements = result
                logger.info("shape at %s: %s (%d elements)", iterator[0].bounds, char, elements)
                if char not in ('T', 'L', 'I', 'l', '-', '/', 'O', 'V', 'U') or last_was_char: # easy for walls to look like these characters
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
                shape_repr += ' or %s' % self._shape_str(shape.child.offset_commands(last_offset)[1])
            if shape.points is not None:
                shape_repr += ' (%s)' % (','.join("%g" % x for x in shape.points.flatten()))
            shape.child.args['title'] = "%s %s" % (path_ids[shape_str], shape_repr)
            last_offset = shape.child.offset

        for count, shape in sorted((v,k) for k,v in paths_by_shape.items()):
            logger.info("%d copies of %s: %s", count, path_ids.get(shape), shape)

    def _mark_text(self, text, chars):
        # FIXME: Some of these might be real
        if len(chars) <= 1:
            return
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

                chars.append(next(iterator))
                text += char.child.vectorchar
                char_bounds = char.bounds

            logger.info('vector text "%s" found at %s: %s', text, first.bounds[:2], chars)
            self._mark_text(text, chars)

    def reify_text(self):
        def _iterator():
            for parent, child, matrix in self.iterelements():
                if not isinstance(child, Group) or 'vectortext' not in child.args.get('class', ''):
                    continue
                bounds = transformed_bounds(child.bounds, matrix)
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
                fontSize='6pt',
                x=center[0],
                y=-center[1],
                center=True,
                valign='middle',
            ))

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MIT floorplans to georeferenced SVGs.')
    parser.add_argument('pdf', help='PDF file to process')
    parser.add_argument('--debug-angle', action='store_true',
                        help='debug north rotation')
    parser.add_argument('--disable-annotations', action='store_true')
    parser.add_argument('--disable-find-text', action='store_true')
    parser.add_argument('--disable-reify-text', action='store_true')

    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('decodegraphics').setLevel(logging.INFO)

    logging.debug("known shapes = %s", KNOWN_SHAPES)

    inpfn = args.pdf
    outfn = 'copy.' + os.path.basename(inpfn)
    pages = PdfReader(inpfn, decompress=True).pages

    parser = Floorplan2Svg()
    parser.debug_angle = args.debug_angle

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
        if not args.disable_find_text:
            parser.find_text()
        annot_mat = np.dot(rotate_mat(rotate/180*np.pi), mat(-1,0,0,1,width,0))
        if parser.north_angle and not parser.debug_angle:
            parser.remove_edge_content(parser.stack[1], IDENTITY)
            parser.stack[1].matrix = np.dot(rotate_mat(parser.north_angle), parser.stack[1].matrix)
            annot_mat = np.dot(rotate_mat(parser.north_angle), annot_mat)
        if not args.disable_reify_text:
            parser.reify_text()
        if annots and not args.disable_annotations:
            annotg = Group(class_="annotations")
            d.append(annotg)
            for a in annots:
                rect = [float(x) for x in a.Rect]
                rect = transformed_points(rect, annot_mat)[:,:2].flatten()
                parser.debug_rect(rect, text=a.Contents.to_unicode(), parent=annotg, stroke="orange")
        print(d.asSvg())

if __name__ == '__main__':
    main()
