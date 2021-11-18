# A part of pdfrw (https://github.com/pmaupin/pdfrw)
# Copyright (C) 2006-2009 Patrick Maupin, Austin, Texas
# MIT license -- See LICENSE.txt for details

'''
This file is an example parser that will parse a graphics stream
into a reportlab canvas.

Needs work on fonts and unicode, but works on a few PDFs.

Better to use Form XObjects for most things (see the example in rl1).

'''
from inspect import getargspec
from itertools import chain

from pdfrw import PdfTokens
from pdfrw.objects import PdfString

class FontInfo(object):
    ''' Pretty basic -- needs a lot of work to work right for all fonts
    '''
    lookup = {
        # WRONG -- have to learn about font stuff...
        'BitstreamVeraSans': 'Helvetica',
        'ArialMT': 'Helvetica',
        'Arial-Black': 'Helvetica',
        'Arial-BoldMT': 'Helvetica',
        'ArialRoundedMTBold': 'Helvetica',
        'ArialNarrow': 'Helvetica',
             }

    def __init__(self, source):
        name = source.BaseFont[1:]
        self.name = self.lookup.get(name, name)
        self.remap = chr
        self.twobyte = False
        info = source.ToUnicode
        if not info:
            return
        info = info.stream.split('beginbfchar')[1].split('endbfchar')[0]
        info = list(PdfTokens(info))
        assert not len(info) & 1
        info2 = []
        for x in info:
            assert x[0] == '<' and x[-1] == '>' and len(x) in (4, 6), x
            i = int(x[1:-1], 16)
            info2.append(i)
        self.remap = dict((x, chr(y)) for (x, y) in
                          zip(info2[::2], info2[1::2])).get
        self.twobyte = len(info[0]) > 4

#############################################################################
# Control structures


def checkname(n):
    assert n.startswith('/')
    return n

def checkarray(a):
    assert isinstance(a, list), a
    return a

def checktext(t):
    assert isinstance(t, PdfString)
    return t

fixparam = dict(f=float, i=int, n=checkname, a=checkarray,
                s=str, t=checktext)

def token(token, params=''):
    def wrap(f):
        f.token = token
        f.paraminfo = params
        if params:
            f.paraminfo = [fixparam[x] for x in params]
        return f
    return wrap

class Meta(type):
    def __new__(cls, name, bases, attr):
        dispatch = dict()
        for obj in attr.values():
            if hasattr(obj, 'token'):
                dispatch[obj.token] = obj
        attr['dispatch'] = dispatch.get
        return type.__new__(cls, name, bases, attr)

class _ParseClass(object, metaclass=Meta):
    @classmethod
    def parsepage(cls, page, canvas=None):
        self = cls()
        contents = page.Contents
        if getattr(contents, 'Filter', None) is not None:
            raise SystemExit('Cannot parse graphics -- page encoded with %s'
                             % contents.Filter)
        if isinstance(contents, list):
            self.tokens = iter(chain.from_iterable(PdfTokens(c.stream) for c in contents))
        else:
            self.tokens = iter(PdfTokens(contents.stream))
        self.params = params = []
        self.canv = canvas
        self.gpath = None
        self.tpath = None
        self.fontdict = dict((x, FontInfo(y)) for
                             (x, y) in page.Resources.Font.items())

        for token in self.tokens:
            func = self.dispatch(token)
            if func is None:
                params.append(token)
                continue
            if func.paraminfo is None:
                func(self, token, ())
                continue
            delta = len(params) - len(func.paraminfo)
            if delta:
                if delta < 0:
                    print ('Operator %s expected %s parameters, got %s' %
                           (token, len(func.paraminfo), params))
                    params[:] = []
                    continue
                else:
                    print ("Unparsed parameters/commands: %s" % params[:delta])
                del params[:delta]
            paraminfo = zip(func.paraminfo, params)
            try:
                params[:] = [x(y) for (x, y) in paraminfo]
            except:
                for i, (x, y) in enumerate(func.paraminfo):
                    try:
                        x(y)
                    except:
                        raise  # For now
                    continue
            print(func.token, params)
            func(self, *params)
            params[:] = []

    #############################################################################
    # Graphics parsing

    @token('[', None)
    def parse_array(self):
        mylist = []
        for token in self.tokens:
            if token == ']':
                break
            mylist.append(token)
        self.params.append(mylist)

    @token('q')
    def parse_savestate(self):
        self.canv.saveState()

    @token('Q')
    def parse_restorestate(self):
        self.canv.restoreState()

    @token('cm', 'ffffff')
    def parse_transform(self, *params):
        self.canv.transform(*params)

    @token('w', 'f')
    def parse_linewidth(self, width):
        self.canv.setLineWidth(width)

    @token('J', 'i')
    def parse_linecap(self, linecap):
        self.canv.setLineCap(linecap)

    @token('j', 'i')
    def parse_linejoin(self, linejoin):
        self.canv.setLineJoin(linejoin)

    @token('M', 'f')
    def parse_miterlimit(self, limit):
        self.canv.setMiterLimit(limit)

    @token('d', 'as')
    def parse_dash(self, array, phase):  # Array, string
        self.canv.setDash(array, phase)

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

    @token('m', 'ff')
    def parse_move(self, x, y):
        if self.gpath is None:
            self.gpath = self.canv.beginPath()
        self.gpath.moveTo(x, y)
        self.current_point = (x, y)

    @token('l', 'ff')
    def parse_line(self, x, y):
        self.gpath.lineTo(x, y)
        self.current_point = (x, y)

    @token('c', 'ffffff')
    def parse_curve(self, *params):
        self.gpath.curveTo(*params)
        self.current_point = params[-2:]

    @token('v', 'ffff')
    def parse_curve1(self, *params): # x2 y2 x3 y3
        parse_curve(self, token, tuple(self.current_point) + tuple(params))

    @token('y', 'ffff')
    def parse_curve2(self, *params): # x1 y1 x3 y3
        parse_curve(self, token, tuple(params) + tuple(params[-2:]))

    @token('h')
    def parse_close(self):
        self.gpath.close()

    @token('re', 'ffff')
    def parse_rect(self, *params):
        if self.gpath is None:
            self.gpath = self.canv.beginPath()
        self.gpath.rect(*params)
        self.current_point = params[-2:]

    @token('S')
    def parse_stroke(self):
        self.finish_path(1, 0, 0)

    @token('s')
    def parse_close_stroke(self):
        self.gpath.close()
        self.finish_path(1, 0, 0)

    @token('f')
    def parse_fill(self):
        self.finish_path(0, 1, 1)

    @token('F')
    def parse_fill_compat(self):
        self.finish_path(0, 1, 1)

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
        self.gpath.close()
        self.finish_path(1, 1, 0)

    @token('b')
    def parse_close_fill_stroke(self):
        self.gpath.close()
        self.finish_path(1, 1, 1)

    @token('n')
    def parse_nop(self):
        self.finish_path(0, 0, 0)


    def finish_path(self, stroke, fill, fillmode):
        if self.gpath is not None:
            canv = self.canv
            canv._fillMode, oldmode = fillmode, canv._fillMode
            canv.drawPath(self.gpath, stroke, fill)
            canv._fillMode = oldmode
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
        self.canv.setStrokeGray(gray)

    @token('g', 'f')
    def parse_fill_gray(self, gray):
        self.canv.setFillGray(gray)

    @token('RG', 'fff')
    def parse_stroke_rgb(self, r, g, b):
        self.canv.setStrokeColorRGB(r, g, b)

    @token('rg', 'fff')
    def parse_fill_rgb(self, r, g, b):
        self.canv.setFillColorRGB(r, g, b)

    @token('K', 'ffff')
    def parse_stroke_cmyk(self, c, m, y, k):
        self.canv.setStrokeColorCMYK(c, m, y, k)

    @token('k', 'ffff')
    def parse_fill_cmyk(self, c, m, y, k):
        self.canv.setFillColorCMYK(c, m, y, k)

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




def debugparser(undisturbed=set('parse_array'.split())):
    def debugdispatch():
        def getvalue(oldval):
            name = oldval[0].__name__

            def myfunc(self, token, params):
                print ('%s called %s(%s)' % (token, name,
                       ', '.join(str(x) for x in params)))
            if name in undisturbed:
                myfunc = oldval[0]
            return myfunc, oldval[1]
        return dict((x, getvalue(y))
                    for (x, y) in _ParseClass.dispatch.items())

    class _DebugParse(_ParseClass):
        dispatch = debugdispatch()

    return _DebugParse.parsepage

parsepage = _ParseClass.parsepage

if __name__ == '__main__':
    import sys
    from pdfrw import PdfReader
    parse = debugparser()
    fname, = sys.argv[1:]
    pdf = PdfReader(fname, decompress=True)
    for i, page in enumerate(pdf.pages):
        print ('\nPage %s ------------------------------------' % i)
        parse(page)