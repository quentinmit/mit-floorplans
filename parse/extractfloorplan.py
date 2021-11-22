#!/usr/bin/env python3

import logging
import sys
import os

import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString

from pdf2svg import Pdf2Svg, Group, token, mat

logger = logging.getLogger('extractfloorplan')

class Floorplan2Svg(Pdf2Svg):
    def stack_matrix(self):
        matrix = mat(1,0,0,1,0,0)
        for element in reversed(self.stack):
            if hasattr(element, 'matrix'):
                matrix = np.dot(element.matrix, matrix)
        logger.debug("stack matrix = %s", matrix)
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
            logger.info("scale = %s", text.to_unicode())

    def finish_path(self, *args):
        if self.gpath is not None:
            bounds = self.gpath.bounds
            logger.info("path %s %s bounds %s", self.gpath.args['d'], self.gpath.transform, self.gpath.bounds)
            maxx, maxy, _ = np.dot((self.gpath.bounds[0], self.gpath.bounds[1], 1), self.stack_matrix())
            logger.info("path bounds %s lowest point on path (%f,%f)", self.gpath.bounds, maxx, maxy)
            if maxy > -0.2*self.top.height:
                self.gpath = None
                return
        super().finish_path(*args)

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
        print(d.asSvg())

if __name__ == '__main__':
    main()
