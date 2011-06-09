#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Q&D random projections stuff to generate fake amber eigen-decomposition matrices.
    Note that this is far from being conservative with memory usage, but should be alright
    with the little residues we are using here.

    Usage: python rp.py evecsfile1 evecsfile2...
    It will generate 5 different normally-distributed random matrices (deterministically).
"""
from __future__ import with_statement
import numpy as np
import os.path as op
import sys

CURRENT_PATH = op.split(op.realpath(__file__))[0]
DEFAULT_EVEC_FILE = op.join(CURRENT_PATH, 'test.evec')

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
    return np.sqrt(np.dot(vector,vector.conj()))

def normalizel2(vec):
    """ Returns a fresh copy of the vector with unit L2-norm """
    return vec / l2norm(vec)

def gen_random_vector(size, rng=np.random, uniform=False):
    return rng.uniform(size=size) if uniform else rng.randn(size)

def gen_random_vectors(num, size, rng=np.random, uniform=False):
    return [gen_random_vector(size, rng, uniform) for _ in range(num)]

def parse_amber_evecs(evecs_file=DEFAULT_EVEC_FILE, evecs_to_numpy=False):
    """ Get the relevant info from the eigenvals file. """
    print 'parsing ' +evecs_file
    with open(evecs_file) as evecs:
        evecs = evecs.read().split(' ****\n')[:-1]
        evals = [float(evec.splitlines()[0].split()[1].strip()) for evec in evecs]
        assert len(evecs) > 0, '%s should have at least one component, but it seems not to have any. Is the format correct?'%evecs_file
        return evals, len(parse_evec(evecs[0])), parse_evecs(evecs) if evecs_to_numpy else None

def one2amber(val, vec, entry_num):
    """ Amber writes the eigenvalue/eigenvector values in entries with the following format:
        Read the code...
    """
    lines = [str(entry_num).rjust(5) + ('%.5f'%val).rjust(12)]
    fullrows = len(vec) / 7 * 7
    for row in vec[:fullrows].reshape([-1,7]):
        lines.append(''.join(map(lambda val: ('%.5f'%val).rjust(11), row)))
    lines.append(''.join(map(lambda val: ('%.5f'%val).rjust(11), vec[fullrows:])))
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
    original = open(DEFAULT_EVEC_FILE).read()
    vector = gen_random_vector(num_atoms_coords)
    assert original == all2amber(evals, evecs)
    assert l2norm(vector) - 1.0 > tol
    assert l2norm(normalizel2(vector)) - 1.0 < tol
    assert l2norm(vector) - 1.0 > tol
    assert are_orthogonal(evecs[0], evecs[1])
    assert l2norm(evecs[0]) - 1.0 < tol
    print 'Everything seems alright'
    print 'Usage: python rp.py evecsfile1 evecsfile2...'

if __name__ == '__main__':
    if len(sys.argv) == 1:
        test()
    for arg in sys.argv[1:]:
        if not op.exists(arg):
            print 'File %s does not exist, skipping...'%arg
        root, name = op.split(arg)
        evals, num_atoms_coords, _ = parse_amber_evecs(arg)  #Err-check this...
        for seed in range(5):
            rng = np.random.RandomState(seed)
            random_vals = sorted(gen_random_vector(len(evals), rng, uniform=True), reverse=True)
            random_vecs = map(normalizel2, gen_random_vectors(len(evals), num_atoms_coords, rng))
            with open(op.join(root, name+'-rp-gaussian-seed_%d.evecs'%seed), 'w', 0) as dest:
                dest.write(all2amber(random_vals, random_vecs))