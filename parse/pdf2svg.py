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

import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString
#import reportlab.pdfgen.canvas
#from reportlab.pdfgen.canvas import Canvas

from decodegraphics import BaseParser, debugparser, token


logger = logging.getLogger('pdf2svg')


def mat(a,b,c,d,e,f):
    return np.array([[a,b,0],[c,d,0],[e,f,1]])


class TransformMixin:
    _matrix = np.identity(3)

    def __init__(self, *args, matrix=None, **kwargs):
        if matrix is not None:
            self._matrix = matrix
            kwargs['transform'] = self.transform
        super().__init__(*args, **kwargs)

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        if matrix is None:
            self._matrix = np.identity(3)
            del self.args['transform']
        else:
            self._matrix = matrix
            self.args['transform'] = self.transform

    @property
    def transform(self):
        if (self.matrix == np.eye(3)).all():
            return None
        return 'matrix(%f,%f,%f,%f,%f,%f)' % tuple(self.matrix[:,:2].flatten())


class Group(TransformMixin, draw.Group):
    @property
    def bounds(self):
        child_bounds = [c.bounds for c in self.children if hasattr(c, 'bounds') and c.bounds is not None]
        if not child_bounds:
            return None
        child_bounds = np.array(child_bounds).reshape((-1, 2))
        child_bounds = np.hstack((child_bounds, np.ones(child_bounds.shape[:-1] + (1,))))
        # matrix of (x,y,1) rows
        # apply transformation first because it can change x and y
        child_bounds = np.dot(child_bounds, self.matrix)
        minx,miny,_ = np.min(child_bounds, axis=0)
        maxx,maxy,_ = np.max(child_bounds, axis=0)
        return (minx,miny,maxx,maxy)

    def __str__(self):
        out = self.__class__.__name__
        if 'class' in self.args:
            out += '.' + self.args['class']
        if self.transform:
            out += ' transform(%s)' % (self.transform)
        return out

class Path(TransformMixin, draw.Path):
    pass

class Text(TransformMixin, draw.Text):
    @property
    def bounds(self):
        if 'x' in self.args and 'y' in self.args:
            # TODO: Calculate width
            x,y = self.args['x'], self.args['y']
            return np.dot([[x,y,1],[x,y,1]], self.matrix)[:,:2].flatten()
        return None


@debugparser()
class Pdf2Svg(BaseParser):
    def parsepage(self, page, top):
        logging.debug("page %s", page)
        logging.debug("resources %s", page.Resources)

        self.top = top
        self.stack = [top]
        self.last = top
        self.gpath = None

        # initialize text state
        self.tmat = None
        self.tc = 0 # charspacing
        self.tw = 0 # wordspacing
        self.th = 1 # horiz scale
        self.tl = 0 # leading
        self.trise = 0 # rise

        if page.Rotate and int(page.Rotate) == 270:
            self.gstate(_matrix=mat(0,-1,1,0,self.top.width,0))
        self.gstate(_matrix=mat(1,0,0,-1,0,0))
        super().parsepage(page)
    #############################################################################
    # Graphics parsing

    def log_stack(self, name):
        out = []
        for element in self.stack:
            out.append(str(element))
        logger.debug("%s() new stack = %s", name, ', '.join(out))

    @token('q')
    def parse_savestate(self, class_='savestate'):
        g = Group(class_=class_)
        self.stack[-1].append(g)
        self.stack.append(g)
        self.last = g
        self.log_stack("savestate")

    @token('Q')
    def parse_restorestate(self, class_='savestate'):
        i = len(self.stack)-1
        while i > 0 and self.stack[i].args.get("class") != class_:
            i -= 1
        if i == 0:
            logging.error("tried to restore state that isn't on the stack")
            return
        self.stack = self.stack[:i]
        self.last = None
        self.log_stack("restorestate")

    def add(self, element):
        self.stack[-1].append(element)
        self.last = element
        logger.debug("added %s to %s", element, self.stack[-1])

    def gstate(self, _matrix=None, **kwargs):
        if not isinstance(self.last, Group):
            #if len(self.stack) > 1:
            #    oldg = self.stack.pop()
            # XXX: close last group and copy settings?
            if isinstance(self.stack[-1], Group) and len(self.stack[-1].args) == 1 and 'transform' in self.stack[-1].args:
                # Special-case bare transformations
                g = self.stack.pop()
                i = self.stack[-1].children.index(g)
                for child in g.children:
                    child.matrix = np.dot(child.matrix, g.matrix)
                # Replace group with individual children
                self.stack[-1].children[i:i+1] = g.children
                _matrix = np.dot(_matrix, g.matrix)
            g = Group()
            self.stack[-1].append(g)
            self.stack.append(g)
            self.last = g
        # XXX: Create new g if overriding the same value (can this happen?)
        for k, v in kwargs.items():
            k = k.replace("_", "-")
            self.last.args[k] = v
        if _matrix is not None:
            if hasattr(self.last, 'matrix'):
                _matrix = np.dot(_matrix, self.last.matrix)
            self.last.matrix = _matrix

    @token('cm', 'ffffff')
    def parse_transform(self, a,b,c,d,e,f):
        self.gstate(_matrix=mat(a,b,c,d,e,f))
        self.stack[-1].matrix

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
            self.gpath = Path()
            # N.B. The path is not drawn until the paint operator, so we can't add it to the tree yet.
            self.gpath.bounds = None

    def touch_point(self, x, y):
        self.current_point = (x, y)
        if not self.gpath.bounds:
            self.gpath.bounds = (x,y,x,y)
        x1,y1,x2,y2 = self.gpath.bounds
        self.gpath.bounds = (min(x, x1), min(y, y1), max(x, x2), max(y, y2))

    @token('m', 'ff')
    def parse_move(self, x, y):
        self.start_path()
        self.gpath.M(x, -y)
        self.touch_point(x, y)

    @token('l', 'ff')
    def parse_line(self, x, y):
        self.gpath.L(x, -y)
        self.touch_point(x, y)

    @token('c', 'ffffff')
    def parse_curve(self, x1, y1, x2, y2, x, y):
        self.gpath.C(x1, -y1, x2, -y2, x, -y)
        self.touch_point(x, y)

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
        assert self.tmat is None
        self.tmat = self.tlmat = mat(1,0,0,1,0,0)
        #self.tpath = Text([], self.curfontsize)
        # XXX: self.curfont.name
        #assert self.tpath

    @token('Tm', 'ffffff')
    def parse_text_transform(self, a,b,c,d,e,f):
        # The matrix is *not* concatenated onto the current text matrix.
        self.tmat = self.tlmat = mat(a,b,c,d,e,f)
        #self.tmat = np.dot(mat(a,b,c,d,e,f), self.tmat)

    @token('Tr', 'i')
    def parse_text_rendering_mode(self, mode):
        # XXX
        pass

    @token('Tf', 'nf')
    def parse_setfont(self, name, size):
        fontinfo = self.fontdict[name]
        #if self.tpath:
        #    self.tpath._setFont(fontinfo.name, size)
        self.curfont = fontinfo
        self.curfontsize = 12#size

    @token('Tj', 't')
    def parse_text_out(self, text):
        logging.info("text %s", text.to_unicode())
        matrix = np.dot(mat(self.curfontsize*self.th,0,0,self.curfontsize,0,self.trise), self.tmat)
        # XXX Update tmat by x += ((w0-(Tj/1000))*tfs+tc+tw)*th
        self.add(
            Text(
                text=text.to_unicode(),
                fontSize=self.curfontsize,
                x=0, y=0, # positioning by transform
                matrix=np.dot(mat(1,0,0,-1,0,0), matrix),
            )
        )

    @token("'", 't')
    def parse_lf_text_out(self, text):
        self.parse_text_line()
        self.parse_text_out(text)

    @token('"', 'fft')
    def parse_lf_text_out_with_spacing(self, wordspace, charspace, text):
        self.parse_set_word_space(wordspace)
        self.parse_set_char_space(charspace)
        self.parse_lf_text_out(text)

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
        self.parse_text_out(text)

    @token('ET')
    def parse_end_text(self):
        assert self.tmat is not None
        self.tmat = None

    @token('Td', 'ff')
    def parse_move_cursor(self, tx, ty):
        self.tmat = self.tlmat = np.dot(mat(1,0,0,1,tx,ty), self.tlmat)

    @token('TL', 'f')
    def parse_set_leading(self, leading):
        self.tl = leading

    @token('T*')
    def parse_text_line(self):
        self.parse_move_cursor(0, self.tl)

    @token('Tc', 'f')
    def parse_set_char_space(self, charspace):
        self.tc = charspace

    @token('Tw', 'f')
    def parse_set_word_space(self, wordspace):
        self.tw = wordspace

    @token('Tz', 'f')
    def parse_set_hscale(self, scale):
        self.th = scale / 100

    @token('Ts', 'f')
    def parse_set_rise(self, rise):
        self.trise = rise

    @token('Do', 'n')
    def parse_xobject(self, name):
        # TODO: Need to do this
        pass

    #############################################################################
    # Tag parsing
    @token('BMC', 'n')
    def parse_begin_marked_content(self, tag):
        logger.debug("begin marked content %s", tag)
        assert self.marked_tag is None
        self.marked_tag = tag[1:]
        self.parse_savestate(class_=self.marked_tag)

    @token('BDC', 'nn')
    def parse_begin_marked_content_props(self, tag, properties):
        logger.debug("begin marked content %s %s", tag, properties)
        assert self.marked_tag is None
        self.marked_tag = tag[1:]
        properties = self.page.Resources.Properties[properties]
        logger.debug(properties)
        if properties.get('/Type') == '/OCG':
            self.current_ocg = properties.Name
            self.parse_savestate(class_=self.marked_tag)
            self.last.args['title'] = self.current_ocg

    @token('EMC')
    def parse_end_marked_content(self):
        logger.debug("end marked content %s", self.marked_tag)
        assert self.marked_tag is not None
        self.parse_restorestate(class_=self.marked_tag)
        self.marked_tag = None
        #if self.current_ocg in ('(A-SHBD)', '(A-VIEW)'):
        #    self.canv.restoreState()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    inpfn, = sys.argv[1:]
    outfn = 'copy.' + os.path.basename(inpfn)
    pages = PdfReader(inpfn, decompress=True).pages

    parser = Pdf2Svg()

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
        print(d.asSvg())

