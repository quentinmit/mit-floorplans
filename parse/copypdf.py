#!/usr/bin/env python

'''
usage:   copy.py my.pdf

Creates copy.my.pdf

Uses somewhat-functional parser.  For better results
for most things, see the Form XObject-based method.

'''

import sys
import os

import reportlab.pdfgen.canvas
#from reportlab.pdfgen.canvas import Canvas

from decodegraphics import Pdf2ReportLab, debugparser
from pdfrw import PdfReader, PdfWriter, PdfArray

inpfn, = sys.argv[1:]
outfn = 'copy.' + os.path.basename(inpfn)
pages = PdfReader(inpfn, decompress=True).pages
canvas = reportlab.pdfgen.canvas.Canvas(outfn, pageCompression=0)

parser = Pdf2ReportLab()

for page in pages:
    box = [float(x) for x in page.MediaBox]
    assert box[0] == box[1] == 0, "demo won't work on this PDF"
    if '/Rotate' in page:
        if int(page.Rotate) == 90 or int(page.Rotate) == 270:
            box[2:] = reversed(box[2:])
        canvas.setPageRotation(int(page.Rotate))
    canvas.setPageSize(box[2:])
    parser.parsepage(page, canvas)
    canvas.showPage()
canvas.save()
