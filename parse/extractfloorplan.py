#!/usr/bin/env python3

import logging
import sys
import os

import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, token, mat, transformed_bounds, transformed_points, IDENTITY

logger = logging.getLogger('extractfloorplan')

KNOWN_SHAPES = {
    "L-1,-4L-3,-5L-6,-7L-10,-7L-13,-5L-14,-4L-16,0L-16,14L15,14L15,0L14,-4L12,-5L9,-7L6,-7L3,-5L2,-4L0,0L0,14": "B",
    "L31,0L31,-10L29,-15L26,-18L23,-19L19,-21L11,-21L7,-19L4,-18L1,-15L0,-10L0,0": "D",
    "L32,0L0,-12L32,-24L0,-24": "M",
    "L32,0L0,-11L32,-23L0,-23": "M",
    "L31,0L0,-12L31,-24L0,-24": "M",
    "L31,0L31,-13L29,-18L28,-19L25,-21L20,-21L17,-19L16,-18L14,-13L14,0": "P",
    "L0,-16L-12,-8L-12,-12L-14,-15L-15,-16L-20,-18L-23,-18L-27,-16L-30,-13L-32,-9L-32,-5L-30,0L-29,1L-26,3": "3",
    "L3,1L4,6L4,9L3,13L-2,16L-9,17L-16,17L-22,16L-25,13L-27,9L-27,7L-25,3L-22,0L-18,-2L-16,-2L-12,0L-9,3L-8,7L-8,9L-9,13L-12,16L-16,17": "6",
    "L-2,4L-5,6L-8,6L-11,4L-12,1L-14,-5L-15,-9L-18,-12L-21,-14L-26,-14L-29,-12L-30,-11L-32,-6L-32,0L-30,4L-29,6L-26,7L-21,7L-18,6L-15,3L-14,-2L-12,-8L-11,-11L-8,-12L-5,-12L-2,-11L0,-6L0,0": "8",
}

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

    def apply(self, predicate, parent=None, matrix=IDENTITY):
        if parent:
            matrix = np.dot(parent.matrix, matrix)
            children = parent.children
        else:
            parent = self.top
            children = parent.elements
        for child in children:
            if isinstance(child, Group):
                self.apply(predicate, child, matrix)
            else:
                predicate(parent, child, matrix)

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
        def f(parent, child, matrix):
            if not isinstance(child, Path):
                return
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

        self.apply(f)

        lines = np.array(lines)

        vectors = lines[:,1,:]-lines[:,0,:]
        logger.debug("north vectors %s", vectors)
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        logger.info("angles = %s", angles*180/np.pi)
        if len(angles):
            self.north_angle = angles[1]

    def find_characters(self):
        def _shape(commands):
            out = []
            for cmd, args in commands:
                out.append(cmd+",".join("%g" % x for x in args))
            return "".join(out)
        path_ids = dict()
        path_ids.update(KNOWN_SHAPES)
        paths_by_shape = dict()
        def f(parent, child, matrix):
            if not isinstance(child, Path):
                return
            bounds = transformed_bounds(child.bounds, matrix)
            commands = tuple(child.offset_commands())[1:]
            shape = _shape(commands)
            paths_by_shape[shape] = paths_by_shape.get(shape, 0) + 1
            logger.info("shape at %s: %s", bounds, shape)
            if shape not in path_ids:
                path_ids[shape] = len(path_ids)
            if shape in KNOWN_SHAPES:
                child.args['stroke'] = 'green'
            child.args['title'] = "%s %s" % (path_ids[shape], shape)
        self.apply(f)
        for count, shape in sorted((v,k) for k,v in paths_by_shape.items()):
            logger.info("%d copies of %s: %s", count, path_ids[shape], shape)

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
