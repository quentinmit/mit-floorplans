#!/usr/bin/env python3

import logging
import sys
import os

import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, Path, token, mat, transformed_bounds, transformed_points, IDENTITY

logger = logging.getLogger('extractfloorplan')

class Floorplan2Svg(Pdf2Svg):
    scale = None
    north_angle = None

    def stack_matrix(self):
        matrix = mat(1,0,0,1,0,0)
        for element in reversed(self.stack):
            if hasattr(element, 'matrix'):
                matrix = np.dot(element.matrix, matrix)
        #logger.debug("stack matrix = %s", matrix)
        return matrix

    def gstate(self, _matrix=None, **kwargs):
        if _matrix is not None:
            new = np.dot(_matrix, self.stack_matrix())
            logger.info("new gstate is at %s, height=%s", new[2], self.top.height)
            if new[2][1] < -0.8*self.top.height:
                g = Group()
                g.matrix = _matrix
                self.stack.append(g)
                self.last = g
                return
        super().gstate(_matrix=_matrix, **kwargs)

    @token('Tj', 't')
    def parse_text_out(self, text):
        super().parse_text_out(text)
        if '" =' in text.to_unicode():
            self.scale = text.to_unicode()
            logger.info("scale = %s", text.to_unicode())

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

    def find_north(self, parent, matrix):
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
        _, _, width, height = box
        rotate = 0
        if '/Rotate' in page:
            rotate = int(page.Rotate)
            if rotate % 180 == 90:
                width, height = height, width
        d = draw.Drawing(width, height)
        parser.parsepage(page, d)
        logger.info("page viewbox = %s", d.viewBox)
        logger.info("%s bounds = %s", parser.stack[1], parser.stack[1].bounds)
        parser.find_north(parser.stack[1], IDENTITY)
        if parser.north_angle:
            parser.remove_edge_content(parser.stack[1], IDENTITY)
            cosA = np.cos(parser.north_angle)
            sinA = np.sin(parser.north_angle)
            rotate_mat = mat(cosA,sinA, -sinA,cosA, 0,0)
            parser.stack[1].matrix = np.dot(rotate_mat, parser.stack[1].matrix)
        print(d.asSvg())

if __name__ == '__main__':
    main()
