#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Random projections stuff and methods to generate fake amber-like eigen-decomposition files.

    Usage as script examples:
      - python rp.py evecsfile1 evecsfile2...
      - python rp.py evecsfile1,1
        Will read evecsfile1, generate only one random projection output file
      - python rp.py 100,200,10
        Will genereate 10 times 100 200-dimensional vectors
      - any of the previous can be combined separated by spaces
    For each input spec, it will generate a number different normally-distributed random matrices
    (with controlled random seed).

    Requires python >= 2.5 and numpy.

    Note that this is far from being conservative with memory usage, but should be alright
    with the small number residues we are using here.
"""
from __future__ import with_statement
import numpy as np
import os.path as op
import sys

CURRENT_PATH = op.split(op.realpath(__file__))[0]
TEST_EVEC_FILE = op.join(CURRENT_PATH, 'test.evec')

def parse_with_zero_negative(val):
    """ AFAIK, python does not parse negative zero as such.
        See: http://en.wikipedia.org/wiki/%E2%88%920_%28number%29
        Just to make easier to write tests comparing with the original input.
    """
    is_neg = val.strip()[0] == '-'
    val = float(val)
    return -0.0 if is_neg and val == 0 else val

def parse_evec(evec):
    """ From a list of strings with numbers separated by spaces, create a numpy array. """
    return np.array(map(parse_with_zero_negative, ' '.join(evec.splitlines()[1:]).split()))

def parse_evecs(evecs):
    """ Return a list of numpy arrays after parsing the different strings """
    return [parse_evec(evec) for evec in evecs]

def are_orthogonal(vector1, vector2, eps=1E-4):
    """ Are two vectors orthogonal? """
    return np.dot(vector1, vector2) < eps

def l2norm(vector):
    """ Do not remember if there is a function in numpy to do this. """
    return np.sqrt(np.dot(vector, vector.conj()))

def normalizel2(vec):
    """ Returns a fresh copy of the vector with unit L2-norm """
    return vec / l2norm(vec)

def gen_random_vector_achlioptas(size, rng=np.random, sparse=False):
    """ Random projection ala Achlioptas 2003 """
    if sparse:
        return 1 - 2 * rng.binomial(1, 0.5, size)
    rv = np.array([1 if r < 1.0 / 6 else 0 if r < 5.0 / 6 else -1 for r in rng.uniform(size)])
    return np.sqrt(3) * rv

def gen_random_vector(size, rng=np.random, uniform=False):
    return rng.uniform(size=size) if uniform else rng.randn(size)

def gen_random_vectors(num, size, rng=np.random, uniform=False):
    return [gen_random_vector(size, rng, uniform) for _ in range(num)]

def parse_amber_evecs(evecs_file=TEST_EVEC_FILE, evecs_to_numpy=False):
    """ Get the relevant info from the eigenvals file. """
    print 'parsing ' + evecs_file
    with open(evecs_file) as evecs:
        evecs = evecs.read().split(' ****\n')[:-1]
        evals = [float(evec.splitlines()[0].split()[1].strip()) for evec in evecs]
        assert len(
            evecs) > 0, '%s should have at least one component, but it seems not to have any. Is the format correct?' % evecs_file
        return evals, len(parse_evec(evecs[0])), parse_evecs(evecs) if evecs_to_numpy else None

def one2amber(val, vec, entry_num):
    """ Amber writes the eigenvalue/eigenvector values in entries with the following format:
    1     3.79020
    0.02512   -0.00180    0.03072    0.02381    0.00409    0.03047    0.03714
    0.00672    0.02933    0.03701    0.01292    0.02791   -0.05276   -0.02159
   -0.00384    0.12654
 ****
    2     3.58386
    0.00266   -0.00434   -0.00305    0.01347   -0.00807    0.00387    0.03029
   -0.00752    0.01879    0.03148   -0.01426    0.02181    0.43497    0.17854
    0.16005   -0.37272
 ****
    This method returns a string with the info in the inputs it that exact format.
    TODO: Doctest this
    """
    lines = [str(entry_num).rjust(5) + ('%.5f' % val).rjust(12)]
    fullrows = len(vec) / 7 * 7
    for row in vec[:fullrows].reshape([-1, 7]):
        lines.append(''.join(map(lambda val: ('%.5f' % val).rjust(11), row)))
    lines.append(''.join(map(lambda val: ('%.5f' % val).rjust(11), vec[fullrows:])))
    lines.append(' ****\n')
    return '\n'.join(lines)

def all2amber(vals, vecs):
    """ Returns a string in amber format for the values and associated vectors """
    reconstructed = []
    for i, (eval, evec) in enumerate(zip(vals, vecs)):
        reconstructed.append(one2amber(eval, evec, i + 1))
    return ''.join(reconstructed)

def test(tol=1E-6):
    """ A few checks """
    evals, num_atoms_coords, evecs = parse_amber_evecs(evecs_to_numpy=True)
    original = open(TEST_EVEC_FILE).read()
    vector = gen_random_vector(num_atoms_coords)
    assert original == all2amber(evals, evecs)
    assert l2norm(vector) - 1.0 > tol
    assert l2norm(normalizel2(vector)) - 1.0 < tol
    assert l2norm(vector) - 1.0 > tol
    assert are_orthogonal(evecs[0], evecs[1])
    assert l2norm(evecs[0]) - 1.0 < tol
    print 'Everything seems alright'
    print """ Usage as script examples:
            - python rp.py evecsfile1 evecsfile2...
            - python rp.py evecsfile1,1
              Will read evecsfile1, generate only one random projection output file
            - python rp.py 100,200,10
              Will genereate 10 times 100 200-dimensional vectors
            - any of the previous can be combined separated by spaces
            For each input spec, it will generate a number different normally-distributed random matrices
            (with controlled random seed)."""

if __name__ == '__main__':

    import traceback
    try:
        import argparse
    except ImportError:
        __import__(op.join(CURRENT_PATH, 'libs', 'argparse.py'))

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def gen_and_save(num_vals, dimensionality, num_files=5, root=None, name=None):
        if not root: root = CURRENT_PATH
        if not name: name = str(num_vals) + '-' + str(dimensionality)
        for seed in range(num_files):
            rng = np.random.RandomState(seed)
            random_vals = sorted(gen_random_vector(num_vals, rng, uniform=True), reverse=True)
            random_vecs = map(normalizel2, gen_random_vectors(num_vals, dimensionality, rng))
            with open(op.join(root, name + '-rp-gaussian-seed=%d.evecs' % seed), 'w', 0) as dest:
                dest.write(all2amber(random_vals, random_vecs))

    if len(sys.argv) == 1:
        test()
    for arg in sys.argv[1:]:
        dims = arg.split(',')
        if 2 <= len(dims) <= 3 and all(map(is_int, dims)):
            if 2 == len(dims): gen_and_save(int(dims[0]), int(dims[1]))
            if 3 == len(dims): gen_and_save(int(dims[0]), int(dims[1]), int(dims[2]))
        else:
            try:
                file_and_num = arg.split(',')
                root, name = op.split(file_and_num[0])
                evals, num_atoms_coords, _ = parse_amber_evecs(file_and_num[0])
                num_files = 5
                if len(file_and_num) == 2:
                    if is_int(file_and_num[1]):
                        num_files = int(file_and_num[1])
                    else:
                        print '%s is not a valid \"file,num_files to generate\" input'%arg
                        print 'trying to generate %d files'%num_files
                gen_and_save(len(evals), num_atoms_coords, num_files=num_files, root=root, name=name)
            except Exception, e:
                traceback.print_exc(file=sys.stderr)
                print e.message
                print 'there has been an error parsing %s, skipping... ' % arg