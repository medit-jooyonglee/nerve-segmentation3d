from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import glob
import sys
import setuptools
import os
print('python modules--------------------------->', glob.glob('src/*.py'))

print('python modules', [os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/*.py')])
__version__ = '0.0.1'


EIGEN_PATH = 'eigen339'
def search_eigen_path():
    # path = os.path.join(library_path, EIGEN_PATH)
    # if os.path.exists(path):
    #     return path
    # else:
    searched = "libs"
    for _ in range(5):
        res =  os.path.join(filepath, searched, EIGEN_PATH)
        if os.path.exists(res):
            res = os.path.realpath(res)
            print('founded eigen path:', res)
            return res
        else:
            searched = '../' + searched

    raise FileExistsError('eigen-path cannot be founded')


# FIXME: set the library path
filepath = os.path.dirname(os.path.realpath(__file__))
# print(filepath)
library_path = os.path.join(filepath, "../../libs")
eigen_path = [
	# os.path.join(library_path, "eigen339")
    search_eigen_path()
]
srcpath = '../'
source_files = [		
]



def deep_search(basepath, filter_ext=['.cpp', '.c']):
    
    searched = []
    for root, dirs, files in os.walk(basepath):
        for file in files:
            for ext in filter_ext:
                if isinstance(ext, str) and file.endswith(ext):
                    # if file.endswith(ext):
                    searched.append(os.path.join(root, file))
                elif isinstance(ext, list):
                    for name_ext in ext:
                        if isinstance(name_ext, str) and file.endswith(name_ext):
                            searched.append(os.path.join(root, file))

    return searched
    
searched_sources = deep_search(os.path.join(filepath, 'source'))
source_files.extend(searched_sources)
    
print("founded source files:")
for file in source_files:
    print(file)
class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(        
		'pyInterpolator',
        [*source_files],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True),
			*eigen_path
			
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11', '-fopenmp']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/openmp'],
        'unix': ['/openmp'],
    }
    l_opts = {
        'msvc': [],
        'unix': ['/openmp'],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

setup(
    name='pyinterpolate',
    version=__version__,
    author='jooyongLee',
    author_email='yong86@dio.co.kr',
    url='',
    description='--- project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.3'],
    setup_requires=['pybind11>=2.3'],
    # cmdclass={'build_ext': BuildExt},
    # packages=find_packages(where='.'),
    packages=['pyinterpolate'],
    # package_dir={'pyinterpolate22': 'pyinterpolate'},
    # py_modules=['pyinterpolate'],
    zip_safe=False,
)
