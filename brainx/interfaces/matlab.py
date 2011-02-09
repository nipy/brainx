""" General matlab interface code """

# Stdlib imports
import os
import re
import tempfile
import subprocess

# Default matlab command
MATLAB_CMD = 'matlab -nojvm -nosplash'

def run_matlab(cmd):
    """Run a single-line matlab command, without creating temporary files."""
    return subprocess.Popen('%s -r \"%s;exit\" ' % (MATLAB_CMD, cmd),
                            stdout=subprocess.PIPE,
                            shell=True).communicate()[0]
    

def run_matlab_script(script_lines, script_name='pyscript'):
    ''' Put multiline matlab script into script file and run '''
    mfile = file(script_name + '.m', 'wt')
    mfile.write(script_lines)
    mfile.close()
    return run_matlab(script_name)


# Functions, classes and other top-level code
class mdict(dict):
    """A dict that validates its input on construction to be matlab-compatible.

    Basically this means:

      - Keys can only be strings that would be valid matlab variable names.
      The rules are basically the same as for Python identifiers, only with the
      single added restriction that leading underscores are forbidden.

      - Values are converted as follows:
        * mdict instances and strings are kept as-is.
        * dicts are recursively constructed into mdict instances.
        * lists of strings (possibly nested) are converted into numpy object
          arrays.
        * numpy object arrays are kept as-is.  They better be valid for
          scipy.io.savemat(), or that will ultimately throw an error.  
        * All types of numbers (ints, floats, numpy arrays of any other type)
         are converted into numpy double-precision arrays.
    """

    def _validate_key(self,k0):
        """Validate that a given key is acceptable.

        The valid key (always a string) is returned."""

        k = str(k0)
        if k.startswith('_'):
            raise ValueError("Invalid key %s starting with underscore" % k)
        # Check that it's a valid python identifier
        exec("%s=1" % k, {})
        return k
   
    def __init__(self,*args,**kw):
        """Construct a new mdict instance.

        The call signature is the same as that of dict(), but the kinds of
        inputs that are accepted are restricted; see class docstring for
        details.
        """
        tmp = dict(*args,**kw)
        for k0,v0 in tmp.items():
            # First, validate the key
            raise NotImplementedError()

def mlab_tempfile(dir=None):
    """Returns a temporary file-like object with valid matlab name.

    The file name is accessible as the .name attribute of the returned object.
    The caller is responsible for closing the returned object, at which time
    the underlying file gets deleted from the filesystem.

    Parameters
    ----------
    
      dir : str
        A path to use as the starting directory.  Note that this directory must
        already exist, it is NOT created if it doesn't (in that case, OSError
        is raised instead).

    Returns
    -------
      f : A file-like object.

    Examples
    --------

    >>> f = mlab_tempfile()
    >>> '-' not in f.name
    True
    >>> f.close()
    """
    valid_name = re.compile(r'^\w+$')

    # Make temp files until we get one whose name is a valid matlab identifier,
    # since matlab imposes that constraint.  Since the temp file routines may
    # return names that aren't valid matlab names, but we can't control that
    # directly, we just keep trying until we get a valid name.  To avoid an
    # infinite loop for some strange reason, we only try 100 times.
    for n in range(100):
        f = tempfile.NamedTemporaryFile(suffix='.m',prefix='tmp_matlab_',
                                        dir=dir)
        # Check the file name for matlab compilance
        fname =  os.path.splitext(os.path.basename(f.name))[0]
        if valid_name.match(fname):
            break
        # Close the temp file we just made if its name is not valid; the
        # tempfile module then takes care of deleting the actual file on disk.
        f.close()
    else:
        raise ValueError("Could not make temp file after 100 tries")
        
    return f
