from setuptools import setup, Extension

functions_module = Extension(
    name='adaptor',
    sources=['adaptor.cpp', 'include/clipper/clipper.cpp'],
    # sources=['adaptor.cpp'],
    include_dirs=[r'.\include',
                  r'D:\install\Anaconda3\include']
)

setup(ext_modules=[functions_module])
