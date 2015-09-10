#!/usr/bin/python

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import input
from future import standard_library
standard_library.install_aliases()
import sys
import dataconverters

fns=sys.argv[1:]

if not fns:
    raise ValueError("No filenames given.")

d=dataconverters.SlocumAscii(fns)
d.add_metadata("Originator",input("Data orginator: "))
d.add_metadata("Dataset",input("Dataset description: "))
d.convert(of=input("Output filename: "))


   
