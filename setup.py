#!/usr/bin/env python
"""Installation script for brainx package.
"""

import os

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists('MANIFEST'): os.remove('MANIFEST')

from setuptools import setup

# Get version and release info, which is all stored in brainx/version.py
execfile(os.path.join('brainx', 'version.py'))

opts = dict(name=name,
            maintainer=maintainer,
            maintainer_email=maintainer_email,
            description=description,
            long_description=long_description,
            url=url,
            download_url=download_url,
            license=license,
            classifiers=classifiers,
            author=author,
            author_email=author_email,
            platforms=platforms,
            version=version,
            packages=packages,
            package_data=package_data,
            install_requires=install_requires,
            tests_require=test_requires
            )

# Only add setuptools-specific flags if the user called for setuptools, but
# otherwise leave it alone
import sys
if 'setuptools' in sys.modules:
    opts['zip_safe'] = False

# Now call the actual setup function
if __name__ == '__main__':
    setup(**opts)
