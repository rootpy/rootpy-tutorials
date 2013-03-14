#!/usr/bin/env python
import argparse
import math

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compute tail p-value of the normal distribution')
parser.add_argument('zvalue', type=float,
                   help='z-value for which you want the corresponding p-value')
parser.add_argument('-m', '--mean', type=float, default=0,
                   help='Mean of the normal distribution')
parser.add_argument('-s', '--sigma', type=float, default=1,
                   help='Witdh (standard deviation) of the normal distribution')
args = parser.parse_args()

# Compute and print pvalue
# Reference: http://en.wikipedia.org/wiki/Normal_distribution
chi = (args.zvalue - args.mean) / args.sigma
pvalue = 0.5 * (1 - math.erf(chi / math.sqrt(2)))
print('p-value: %s' % pvalue)
