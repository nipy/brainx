"""Some utilities for manipulating recarrays.

Warning
-------

This module should *never* be imported as 'import *'
"""

import numpy as np
import numpy.testing as nt


# XXX - It's probably OK to import something, but for now let's ban * imports
# altogether .
__all__ = []

#-----------------------------------------------------------------------------
# Functions and public utilities
#-----------------------------------------------------------------------------

def extrude(arr,flatten=False):
    """Create a view of a recarray with one extra 'extruded' dimension.

    XXX - document more...    
    """

    dt = arr.dtype

    fieldtypes = [ v[0] for v in dt.fields.values() ]
    
    if len(set(fieldtypes)) > 1:
        raise ValueError("dtype of recarray must be uniform")
    newdtype = fieldtypes[0]

    nfields = len(dt.fields)

    # If axis is None, for a normal array this means flatten everything and
    # return a single number. In our case, we actually want to keep the last
    # dimension (the "extruded" one) alive so that we can reconstruct the
    # recarray in the end.
    if flatten:
        newshape = (arr.size,nfields)
    else:
        newshape = arr.shape + (nfields,)
    
    # Make the new temp array we'll work with
    return np.reshape(arr.view(newdtype),newshape)


def intrude(arr,dtype):
    """Intrude a recarray by 'flattening' its last dimension into a composite
    dtype.

    XXX - finish doc
    """
    outshape = arr.shape[:-1]
    return (np.reshape(arr.view(dtype),outshape)).view(np.recarray)


def offset_axis(axis):
    """Axis handling logic that is generic to all reductions."""
    flatten = axis is None
    if flatten:
        axis = 0
    else:
        if axis < 0:
            # The case of a negative input axis needs compensation, because we
            # are adding a dimension by ourselves
            axis -= 1
    return flatten, axis

    
def reduction_factory(name):
    """Create a reduction operation for a given method name.
    """
    def op(arr, axis=None):
        # XXX what the hell is this logic?
        flatten, axis = offset_axis(axis)
        
        newarr = extrude(arr,flatten)
        # Do the operation on the new array
        method = getattr(newarr,name)
        result = method(axis)
        # Make the output back into a recarray of the original dtype
        return intrude(result, arr.dtype)

    doc = "%s of a recarray, preserving its structure." % name
    op.__doc__ = doc
    op.func_name = name
    return op


# For methods in the array interface that take an axis argument, the pattern is
# always the same: extrude, operate, intrude.  So we just auto-generate these
# functions here.
reduction_names = ['mean', 'std', 'var', 'min', 'max',
                   'sum', 'cumsum', 'prod', 'cumprod' ]

for fname in reduction_names:
    exec "%s = reduction_factory('%s')" % (fname, fname)

def binop_factory(func):
    """Create a binary operation for a given name.
    """
    def op(a1, a2, out=None):

        new_a1 = extrude(a1)
        new_a2 = extrude(a2)
        if out is not None:
            out = extrude(out)
            
        # Do the operation on the new array
        if out is None:
            result = func(new_a1, new_a2)
        else:
            result = func(new_a1, new_a2, out)
        # Make the output back into a recarray of the original dtype
        return intrude(result, a1.dtype)

    doc = "Binary %s of two recarrays, preserving their structure." % name
    op.__doc__ = doc
    op.func_name = name
    return op


# For methods in the array interface that take an axis argument, the pattern is
# always the same: extrude, operate, intrude.  So we just auto-generate these
# functions here.
binops = [('add',np.add), ('subtract',np.subtract), ('multiply',np.multiply), ('divide',np.divide) ]
#binops = [('add',np.add), np.subtract, np.multiply, np.divide ]
for name, func in binops:
    exec "%s = binop_factory(func)" % name


#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

def test_mean_zero():
    dt = np.dtype(dict(names=['x','y'], formats=[float,float]))
    z = np.zeros((2,3), dt)
    nt.assert_equal(mean(z),z)
    return 1


def mk_xyz():
    """Test utility, make x, y, z arrays."""
    dt = np.dtype(dict(names=['x','y'],formats=[float,float]))
    x = np.arange(6,dtype=float).reshape(2,3)
    y = np.arange(10,16,dtype=float).reshape(2,3)
    z = np.empty( (2,3), dt).view(np.recarray)
    z.x = x
    z.y = y
    return x, y, z


def mk_xyzw():
    """Test utility, make x, y, z, w arrays."""
    x, y, z = mk_xyz()
    w = z.copy()
    w.x *= 2
    w.y *= 2
    return x, y, z, w


def test_reductions():
    x, y, z = mk_xyz()
    for fname in reduction_names:
        reduction = eval(fname)
        xmeth = getattr(x, fname)
        ymeth = getattr(y, fname)
        for axis in [None,0,1,-1,-2]:
            zred = reduction(z,axis)
            yield(nt.assert_equal, zred.x, xmeth(axis))
            yield(nt.assert_equal, zred.y, ymeth(axis))


def test_binops():
    x, y, z, w = mk_xyzw()
    for fname in binop_names:
        op = eval(fname)
        npop = getattr(np, fname)
        opres = op(z,w)
        yield(nt.assert_equal, opres.x, npop(z.x, w.x) )
        yield(nt.assert_equal, opres.y, npop(z.y, w.y) )

# Test support utilities

def eval_tests(testgen):
    """Little utility to consume a nose-compliant test generator.

    Returns
    -------
    The number of executed tests.  An exception is raised if any fails."""
    return len([ t[0](*t[1:]) for t in testgen() ])

# Mark it as not being a test itself, so nose doesn't try to run it
eval_tests.__test__ = False


def run_test_suite():
    """Call all our tests in sequence.

    This lets us run the script as a test suite without needing nose or any
    other test runner for simple cases"""
    from  time  import clock

    # Initialize counters
    ntests = 0
    start = clock()
    
    # Call the tests and count them
    ntests += test_mean_zero()
    ntests += eval_tests(test_reductions)
    ntests += eval_tests(test_binops)

    # Stop clock and summarize
    end = clock()
    print '-'*70
    print "Ran %s tests in %.3f" % (ntests, end-start)
    print '\nOK'
    
run_test_suite.__test__ = False


# If run as a script, just run all the tests and print summary if successful
if __name__ == '__main__':
    run_test_suite()
