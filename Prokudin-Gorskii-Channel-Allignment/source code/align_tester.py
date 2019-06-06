#!/usr/bin/python

from sys import argv, stdout, exit
from os import mkdir
from os.path import dirname, exists
from align import align
from skimage.io import imread, imsave
import csv
import time

if len(argv) != 3:
    stdout.write('Usage: %s input_dir output_dir\n' % argv[0])
    exit(1)

input_dir = argv[1]
output_dir = argv[2]

with open(input_dir + '/gt.csv') as fhandle:
    reader = csv.reader(fhandle)
    # Skip header
    next(reader)

    for row in reader:
        start = time.time()
        filename = row[0]
        b_row, b_col, g_row, g_col, r_row, r_col, diff_max = map(int, row[1:])
        img = imread(input_dir + '/' + filename, plugin='matplotlib')
        img, b_coord, r_coord = align(img, (g_row, g_col))
        diff = abs(b_row - b_coord[0]) + abs(b_col - b_coord[1]) + \
                abs(r_row - r_coord[0]) + abs(r_col - r_coord[1])

        if diff <= diff_max:
            res = 'OK'
        else:
            res = 'FAIL'

        print(filename, res)
        print(b_row, b_col, b_coord)
        print(r_row, r_col, r_coord)

        out_filename = output_dir + '/' + filename
        out_dirname = dirname(out_filename)
        if not exists(out_dirname):
            mkdir(out_dirname)
        imsave(out_filename, img)

        print(time.time() - start)
        print()