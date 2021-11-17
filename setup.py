#!/usr/bin/env python
NAME = 'Iaji'
AUTHOR = 'Iyad Suleiman'
AUTHOR_EMAIL = 'isule@fysik.dtu.dk'
LICENSE = 'LGPLv3'
URL = ''
DOWNLOAD_URL = ''
DESCRIPTION = 'Utility packages of various types'
LONG_DESCRIPTION = '''\
Collection of instruments drivers for instruments available at qpit
'''
CLASSIFIERS = [
    'Development Status :: 1 - beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development :: Libraries',
    'Topic :: System :: Hardware',
    'Operating System :: Linux :: Ubuntu'
]
PLATFORMS = ['all']
MAJOR               = 0
MINOR               = 1
ISRELEASED          = False
VERSION             = '%d.%d' % (MAJOR, MINOR)

if __name__=='__main__':

    from setuptools import setup

    setup(
        name = NAME,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        license = LICENSE,
        url = URL,
        download_url = DOWNLOAD_URL,
        version = VERSION,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        classifiers = CLASSIFIERS,
        platforms = PLATFORMS,
        packages = ['Iaji'])
