#!/usr/bin/env python

'''
usage:   copy.py my.pdf

Creates copy.my.pdf

Uses somewhat-functional parser.  For better results
for most things, see the Form XObject-based method.

'''

import logging
import sys
import os

import drawSvg as draw
#import reportlab.pdfgen.canvas
#from reportlab.pdfgen.canvas import Canvas

from decodegraphics import BaseParser, debugparser, token
from pdfrw import PdfReader, PdfWriter, PdfArray


logger = logging.getLogger('pdf2svg')


@debugparser()
class Pdf2Svg(BaseParser):
    def parsepage(self, page, top):
        self.top = top
        self.stack = [top]
        self.last = top
        self.gpath = None
        super().parsepage(page)
    #############################################################################
    # Graphics parsing

    @token('q')
    def parse_savestate(self):
        g = draw.Group(class_='savestate')
        self.stack[-1].append(g)
        self.stack.append(g)
        self.last = g
        logger.debug("savestate() new stack = %s", self.stack)

    @token('Q')
    def parse_restorestate(self):
        while self.stack[-1].args.get("class") != 'savestate':
            self.stack.pop()
        self.stack.pop()
        self.last = None
        logger.debug("restorestate() new stack = %s", self.stack)

    def add(self, element):
        self.stack[-1].append(element)
        self.last = element
        logger.debug("added %s to %s", element, self.stack[-1])

    def gstate(self, **kwargs):
        if not isinstance(self.last, draw.Group):
            #if len(self.stack) > 1:
            #    oldg = self.stack.pop()
            # XXX: close last group and copy settings?
            g = draw.Group()
            self.stack[-1].append(g)
            self.stack.append(g)
            self.last = g
        # XXX: Create new g if overriding the same value (can this happen?)
        for k, v in kwargs.items():
            self.last.args[k] = v

    @token('cm', 'ffffff')
    def parse_transform(self, *params):
        self.gstate(transform='matrix(%f,%f,%f,%f,%f,%f)' % params)

    @token('w', 'f')
    def parse_linewidth(self, width):
        self.gstate(stroke_width=width)

    @token('J', 'i')
    def parse_linecap(self, linecap):
        self.gstate(stroke_linecap=[
            'butt',
            'round',
            'square',
            ][linecap])

    @token('j', 'i')
    def parse_linejoin(self, linejoin):
        self.gstate(stroke_linejoin=[
            'miter',
            'round',
            'bevel',
            ][linejoin])

    @token('M', 'f')
    def parse_miterlimit(self, limit):
        self.gstate(stroke_miterlimit=limit)

    @token('d', 'as')
    def parse_dash(self, array, phase):  # Array, string
        if not array:
            self.gstate(stroke_dasharray='none')
            return
        self.gstate(
            stroke_dasharray=','.join(array),
            stroke_dashoffset=phase,
        )

    @token('ri', 'n')
    def parse_intent(self, intent):
        # TODO: add logging
        pass

    @token('i', 'i')
    def parse_flatness(self, flatness):
        # TODO: add logging
        pass

    @token('gs', 'n')
    def parse_gstate(self, dictname):
        # TODO: add logging
        # Could parse stuff we care about from here later
        pass

    def start_path(self):
        if self.gpath is None:
            self.gpath = draw.Path()
            # N.B. The path is not drawn until the paint operator, so we can't add it to the tree yet.

    @token('m', 'ff')
    def parse_move(self, x, y):
        self.start_path()
        self.gpath.M(x, y)
        self.current_point = (x, y)

    @token('l', 'ff')
    def parse_line(self, x, y):
        self.gpath.L(x, y)
        self.current_point = (x, y)

    @token('c', 'ffffff')
    def parse_curve(self, x1, y1, x2, y2, x, y):
        self.gpath.C(x1, y1, x2, y2, x, y)
        self.current_point = (x, y)

    @token('v', 'ffff')
    def parse_curve1(self, x2, y2, x, y):
        self.parse_curve(self, self.current_point[0], self.current_point[1], x2, y2, x, y)

    @token('y', 'ffff')
    def parse_curve2(self, x1, y1, x, y): # x1 y1 x3 y3
        self.parse_curve(self, x1, y1, x, y, x, y)

    @token('h')
    def parse_close(self):
        self.gpath.Z()

    @token('re', 'ffff')
    def parse_rect(self, x, y, width, height):
        self.parse_move(x, y)
        self.parse_line(x+width, y)
        self.parse_line(x+width, y+height)
        self.parse_line(x, y+height)
        self.parse_close()
        self.current_point = None

    @token('S')
    def parse_stroke(self):
        self.finish_path(1, 0, 0)

    @token('s')
    def parse_close_stroke(self):
        self.parse_close()
        self.finish_path(1, 0, 0)

    @token('f')
    def parse_fill(self):
        self.finish_path(0, 1, 1)

    @token('F')
    def parse_fill_compat(self):
        self.parse_fill()

    @token('f*')
    def parse_fill_even_odd(self):
        self.finish_path(0, 1, 0)

    @token('B*')
    def parse_fill_stroke_even_odd(self):
        self.finish_path(1, 1, 0)

    @token('B')
    def parse_fill_stroke(self):
        self.finish_path(1, 1, 1)

    @token('b*')
    def parse_close_fill_stroke_even_odd(self):
        self.parse_close()
        self.finish_path(1, 1, 0)

    @token('b')
    def parse_close_fill_stroke(self):
        self.parse_close()
        self.finish_path(1, 1, 1)

    @token('n')
    def parse_nop(self):
        self.finish_path(0, 0, 0)


    def finish_path(self, stroke, fill, fillmode):
        if self.gpath is not None:
            self.gpath.args["fill-rule"] = [
                'evenodd',
                'nonzero',
                ][fillmode]
            # XXX: Default stroke and fill?
            if not stroke:
                self.gpath.args["stroke"] = "none"
            if not fill:
                self.gpath.args["fill"] = "none"
            self.add(self.gpath)
            self.gpath = None

    @token('W')
    def parse_clip_path(self):
        # TODO: add logging
        pass

    @token('W*')
    def parse_clip_path_even_odd(self):
        # TODO: add logging
        pass

    @token('G', 'f')
    def parse_stroke_gray(self, gray):
        self.gstate(stroke="hsl(0, 0%%, %f%%)" % (100*gray))

    @token('g', 'f')
    def parse_fill_gray(self, gray):
        self.gstate(fill="hsl(0, 0%%, %f%%)" % (100*gray))

    @token('RG', 'fff')
    def parse_stroke_rgb(self, r, g, b):
        self.gstate(stroke="rgb(%f%%, %f%%, %f%%)" % (100*r, 100*g, 100*b))

    @token('rg', 'fff')
    def parse_fill_rgb(self, r, g, b):
        self.gstate(fill="rgb(%f%%, %f%%, %f%%)" % (100*r, 100*g, 100*b))

    @token('K', 'ffff')
    def parse_stroke_cmyk(self, c, m, y, k):
        assert False

    @token('k', 'ffff')
    def parse_fill_cmyk(self, c, m, y, k):
        assert False

    #############################################################################
    # Text parsing

    @token('BT')
    def parse_begin_text(self):
        assert self.tpath is None
        self.tpath = self.canv.beginText()
        self.tpath._setFont(self.curfont.name, self.curfontsize)
        assert self.tpath

    @token('Tm', 'ffffff')
    def parse_text_transform(self, *matrix):
        path = self.tpath

        # Stoopid optimization to remove nop
        try:
            code = path._code
        except AttributeError:
            pass
        else:
            if code[-1] == '1 0 0 1 0 0 Tm':
                code.pop()

        path.setTextTransform(*matrix)

    @token('Tr', 'i')
    def parse_text_rendering_mode(self, mode):
        pass

    @token('Tf', 'nf')
    def parse_setfont(self, name, size):
        fontinfo = self.fontdict[name]
        if self.tpath:
            self.tpath._setFont(fontinfo.name, size)
        self.curfont = fontinfo
        self.curfontsize = size

    @token('Tj', 't')
    def parse_text_out(self, text):
        #text = text.decode(self.curfont.remap, self.curfont.twobyte)
        print("text", text.to_unicode())
        self.tpath.textOut(text.to_unicode())

    @token("'", 't')
    def parse_lf_text_out(self, text):
        self.tpath.textLine()
        self.parse_text_out(text)

    @token('"', 'fft')
    def parse_lf_text_out_with_spacing(self, wordspace, charspace, text):
        self.tpath.setWordSpace(wordspace)
        self.tpath.setCharSpace(charspace)
        self.tpath.textLine()
        self.parse_text_out(text)

    @token('TJ', 'a')
    def parse_TJ(self, array):
        remap = self.curfont.remap
        twobyte = self.curfont.twobyte
        result = []
        for x in array:
            if isinstance(x, PdfString):
                #result.append(x.decode(remap, twobyte))
                result.append(x.to_unicode())
            else:
                # TODO: Adjust spacing between characters here
                int(x)
        text = ''.join(result)
        self.tpath.textOut(text)

    @token('ET')
    def parse_end_text(self):
        assert self.tpath is not None
        self.canv.drawText(self.tpath)
        self.tpath = None

    @token('Td', 'ff')
    def parse_move_cursor(self, tx, ty):
        self.tpath.moveCursor(tx, -ty)

    @token('TL', 'f')
    def parse_set_leading(self, leading):
        self.tpath.setLeading(leading)

    @token('T*')
    def parse_text_line(self):
        self.tpath.textLine()

    @token('Tc', 'f')
    def parse_set_char_space(self, charspace):
        self.tpath.setCharSpace(charspace)

    @token('Tw', 'f')
    def parse_set_word_space(self, wordspace):
        self.tpath.setWordSpace(wordspace)

    @token('Tz', 'f')
    def parse_set_hscale(self, scale):
        self.tpath.setHorizScale(scale - 100)

    @token('Ts', 'f')
    def parse_set_rise(self, rise):
        self.tpath.setRise(rise)

    @token('Do', 'n')
    def parse_xobject(self, name):
        # TODO: Need to do this
        pass


logging.basicConfig(level=logging.DEBUG)

inpfn, = sys.argv[1:]
outfn = 'copy.' + os.path.basename(inpfn)
pages = PdfReader(inpfn, decompress=True).pages

parser = Pdf2Svg()

sys.setrecursionlimit(sys.getrecursionlimit()*2)

for page in pages:
    box = [float(x) for x in page.MediaBox]
    assert box[0] == box[1] == 0, "demo won't work on this PDF"
    d = draw.Drawing(box[2], box[3])
    if '/Rotate' in page:
        #if int(page.Rotate) == 90 or int(page.Rotate) == 270:
        #    box[2:] = reversed(box[2:])
        if int(page.Rotate) != 0:
            d.svgArgs["transform"] = "rotate(%d)" % int(page.Rotate)
    parser.parsepage(page, d)
    print(d.asSvg())

