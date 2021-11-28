#!/usr/bin/env python

'''
usage:   copy.py my.pdf

Creates copy.my.pdf

Uses somewhat-functional parser.  For better results
for most things, see the Form XObject-based method.

'''

from collections import namedtuple
import functools
import itertools
import logging
import sys
import os
import re
from typing import GenericAlias

import numpy as np
import drawSvg as draw
from pdfrw import PdfReader, PdfWriter, PdfArray, PdfString
import bezier
#import reportlab.pdfgen.canvas
#from reportlab.pdfgen.canvas import Canvas

from decodegraphics import BaseParser, debugparser, token


logger = logging.getLogger('pdf2svg')


def mat(a,b,c,d,e,f):
    return np.array([[a,b,0],[c,d,0],[e,f,1]])

def rotate_mat(angle_radians):
    cosA = np.cos(angle_radians)
    sinA = np.sin(angle_radians)
    return mat(cosA,sinA, -sinA,cosA, 0,0)


IDENTITY = np.identity(3)


def transformed_points(points, matrix):
    points = np.array(points).reshape((-1, 2))
    points = np.hstack((points, np.ones(points.shape[:-1] + (1,))))
    # matrix of (x,y,1) rows
    # apply transformation first because it can change x and y
    points = np.dot(points, matrix)
    return points


def transformed_bounds(points, matrix):
    if points is None or len(points) == 0:
        return None
    points = np.array(points).reshape((-1, 2))
    if (matrix == IDENTITY).all():
        minx = min(x for x, _ in points)
        miny = min(y for _, y in points)
        maxx = max(x for x, _ in points)
        maxy = max(y for _, y in points)
        return Bounds(minx,miny,maxx,maxy)
    points = transformed_points(points, matrix)
    minx,miny,_ = np.min(points, axis=0)
    maxx,maxy,_ = np.max(points, axis=0)
    #logger.debug("bounds of %s = %s", self, (minx,miny,maxx,maxy))
    return Bounds(minx,miny,maxx,maxy)


class Bounds(namedtuple("Bounds", "minx miny maxx maxy".split())):
    @property
    def width(self):
        return self.maxx-self.minx

    @property
    def height(self):
        return self.maxy-self-miny


class TransformMixin:
    _matrix = IDENTITY

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
            self._matrix = IDENTITY
            del self.args['transform']
        else:
            self._matrix = matrix
            self.args['transform'] = self.transform

    def apply_matrix(self, matrix):
        self.matrix = np.dot(self.matrix, matrix)

    @property
    def transform(self):
        if (self.matrix == IDENTITY).all():
            return None
        return 'matrix(%f,%f,%f,%f,%f,%f)' % tuple(self.matrix[:,:2].flatten())

    def transformed_bounds(self, points):
        return transformed_bounds(points, self.matrix)


class Group(TransformMixin, draw.Group):
    current_offset = (0,0)
    @property
    def bounds(self):
        child_bounds = [c.bounds for c in self.children]
        child_bounds = [b for b in child_bounds if b is not None]
        if not child_bounds:
            return None
        child_bounds = np.array(child_bounds).reshape((-1, 2))
        #logger.info("%s child_bounds = %s", self, child_bounds)
        return self.transformed_bounds(child_bounds)

    def projected_bounds(self, matrix=IDENTITY):
        child_bounds = [c.projected_bounds(np.dot(self.matrix, matrix)) for c in self.children]
        child_bounds = [b for b in child_bounds if b is not None]
        if not child_bounds:
            return None
        child_bounds = np.array(child_bounds).reshape((-1, 2))
        return transformed_bounds(child_bounds, IDENTITY)

    def __str__(self):
        out = self.__class__.__name__
        if 'class' in self.args:
            out += '.' + self.args['class']
        if self.transform:
            out += ' transform(%s)' % (self.transform)
        return out

class path_property:
    def __init__(self, func):
        self.func = func
        self.attrname = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name):
        self.attrname = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        cache = instance.__dict__
        (d, val) = cache.get(self.attrname, (None, None))
        if val is None or instance.args["d"] != d:
            val = self.func(instance)
            cache[self.attrname] = (instance.args["d"], val)
        return val

    def __set__(self, instance, value):
        raise AttributeError()

    __class_getitem__ = classmethod(GenericAlias)


class Curve(bezier.Curve):
    def __repr__(self):
        return 'Curve.from_nodes(%r.T)' % (self.nodes.T.tolist(),)

_PATH_SPLIT_RE = re.compile(r"(?<!^) *(?=[a-zA-Z])")


class Path(TransformMixin, draw.Path):
    @path_property
    def _points(self):
        points = []
        for command in self.args['d'].split(' '):
            args = []
            if len(command) > 1:
                args = [float(x) for x in command[1:].split(',')]
            cmd = command[0]
            if points:
                cpx, cpy = points[-1]
            if cmd == 'H':
                points.append((args[0], cpy))
            elif cmd == 'h':
                points.append((cpx+args[0], cpy))
            elif cmd == 'V':
                points.append((cpx, args[0]))
            elif cmd == 'v':
                points.append((cpx, cpy+args[0]))
            elif len(args) > 1:
                if cmd.isupper():
                    points.append((args[-2], args[-1]))
                else:
                    points.append((cpx+args[-2], cpy+args[-2]))
        return points

    @path_property
    def bounds(self):
        return self.transformed_bounds(np.array(self._points))

    def projected_bounds(self, matrix):
        return transformed_bounds(self._points, np.dot(self.matrix, matrix))

    @path_property
    def commands(self):
        try:
            return [(c[0], [float(x) for x in c[1:].split(',') if x != '']) for c in _PATH_SPLIT_RE.split(self.args['d']) if c]
        except IndexError:
            logger.exception("parsing path %s", self.args["d"])
            raise

    @path_property
    def offset(self):
        if self.commands[0][0] != 'M':
            return (0,0)
        else:
            return self.commands[0][1]

    def offset_commands(self, offset=None):
        commands = list(self.commands) # make a copy
        if not commands:
            return (offset or (0,0)), commands
        if not offset:
            if commands[0][0] != 'M':
                raise ValueError("cannot determine offset for path; first command was %s", commands[0])
            offset = commands[0][1]
            commands = commands[1:]
        x,y = offset
        for i, (cmd, args) in enumerate(commands):
            if cmd == 'H':
                args = (args[0]-x,)
            elif cmd == 'V':
                args = (args[0]-y,)
            elif cmd.isupper():
                args = tuple(a-b for a,b in zip(args, itertools.cycle((x, y))))
            commands[i] = (cmd, args)
        return offset, tuple(commands)

    @path_property
    def curves(self):
        _, commands = self.offset_commands()
        curves = []
        initial = last = (0., 0.)
        def _add(*points):
            curves.append(Curve.from_nodes(np.array(points).T))
        for (cmd, args) in commands:
            if cmd == 'M':
                initial = last = args
            elif cmd == 'L':
                _add(last, last := args)
            elif cmd == 'H':
                _add(last, last := (args[0], last[1]))
            elif cmd == 'V':
                _add(last, last := (last[0], args[0]))
            elif cmd == 'C':
                while args:
                    _add(last, args[0:2], args[2:4], last := args[4:6])
                    args = args[6:]
            elif cmd in ('Z', 'z'):
                _add(last, last := initial)
        return curves

    @path_property
    def length(self):
        return sum(c.length for c in self.curves)

    def quantize(self, points=32):
        lengths = np.array([0] + [c.length for c in self.curves])
        total_length = np.sum(lengths)
        if total_length == 0:
            return np.zeros((points, 2))
        offsets = np.interp(
            np.linspace(0, total_length, points),
            np.cumsum(lengths),
            np.arange(lengths.size),
        )
        splits = np.unique(offsets.astype(int), return_index=True)[1][1:-1]
        #logger.debug("quantizing %s with curves of length %s", self, lengths)
        #logger.debug("calculated offsets: %s", offsets)
        #logger.debug("split points: %s", splits)
        points = []
        for offsets in np.split(offsets, splits):
            i = int(offsets[0])
            curve = self.curves[i]
            points.append(curve.evaluate_multi(offsets-i).T)
        return np.vstack(points)

    def __repr__(self):
        return 'Path(d="%s")' % (self.args['d'])


class Text(TransformMixin, draw.Text):
    @property
    def bounds(self):
        return self.projected_bounds()

    def projected_bounds(self, matrix=IDENTITY):
        if 'x' in self.args and 'y' in self.args:
            # TODO: Calculate width
            x,y = self.args['x'], self.args['y']
            return Bounds(*np.dot([[x,y,1],[x,y,1]], np.dot(self.matrix, matrix))[:,:2].flatten())
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

        # PDF coordinate system is (0,0) at bottom left with positive y going up the screen
        # SVG coordinate system is (0,0) at top left with positive y going down the screen
        # So, apply a transformation matrix to flip the y axis
        # Setting the viewBox accordingly is the responsibility of the caller.
        self.pdf2svg = Group(class_="pdf2svg", matrix=mat(1,0,0,-1,0,0))
        self.add(self.pdf2svg)
        self.stack.append(self.pdf2svg)

        if page.Rotate and int(page.Rotate) == 270:
            self.pdf2svg.apply_matrix(mat(0,-1,1,0,self.top.width,0))
        # FIXME: Why do we not need to shift by the height of the page to fix (0,0)?

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
        g.current_offset = self.stack[-1].current_offset
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
        #logger.debug("added %s to %s", element, self.stack[-1])

    def gstate(self, _matrix=None, **kwargs):
        if _matrix is not None:
            if (_matrix[:2] == [[1,0,0],[0,1,0]]).all():
                # Pure offset
                self.stack[-1].current_offset = np.add(self.stack[-1].current_offset, _matrix[2,0:2])
                _matrix = None
            elif hasattr(self.stack[-1], 'current_offset'):
                x,y = self.stack[-1].current_offset
                _matrix = np.dot(_matrix, mat(1,0,0,1,x,y))
        if _matrix is None and len(kwargs) == 0:
            return
        if not isinstance(self.last, Group) or self.last == self.pdf2svg:
            #if len(self.stack) > 2:
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
            if isinstance(self.stack[-1], Group):
                s1 = set(self.stack[-1].args.keys())
                s2 = set(k.replace("_", "-") for k in kwargs.keys())
                logger.debug("s1 = %s, s2 = %s", s1, s2)
                if s1 == s2:
                    # Assume all properties are overrides.
                    # Not safe for CSS in general, but we never use relative values.
                    self.stack.pop()
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

    def offset(self, x, y):
        return np.add(self.stack[-1].current_offset, (x,y))

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

    def touch_point(self, x, y):
        self.current_point = (x, y)

    @token('m', 'ff')
    def parse_move(self, x, y):
        x,y = self.offset(x,y)
        self.start_path()
        self.gpath.M(x, -y)
        self.touch_point(x, y)

    @token('l', 'ff')
    def parse_line(self, x, y):
        x,y = self.offset(x,y)
        self.gpath.L(x, -y)
        self.touch_point(x, y)

    @token('c', 'ffffff')
    def parse_curve(self, x1, y1, x2, y2, x, y):
        x1,y1 = self.offset(x1,y1)
        x2,y2 = self.offset(x2,y2)
        x,y = self.offset(x,y)
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
            if isinstance(self.last, Path):
                a1 = {k:v for k,v in self.gpath.args.items() if k != 'd'}
                a2 = {k:v for k,v in self.last.args.items() if k != 'd'}
                if a1 == a2 and len(self.last.args['d']) < 1024 and self.gpath.args['d'][0] == 'M' and False:
                    #logger.debug('extending %s', self.last)
                    self.last.args['d'] += ' ' + self.gpath.args['d']
                    return
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
        if isinstance(text, PdfString):
            text = text.to_unicode()
        logging.info("text %s", text)
        matrix = np.dot(mat(self.curfontsize*self.th,0,0,self.curfontsize,0,self.trise), self.tmat)
        # XXX Update tmat by x += ((w0-(Tj/1000))*tfs+tc+tw)*th
        self.add(
            Text(
                text=text,
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
                float(x)
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

