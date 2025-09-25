# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

import sys
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by MindSpeed-LLM.")

__description__ = 'MindSpeed-LLM for LLMs of Ascend'
__version__ = '0.0.1'
__author__ = 'Ascend'
__long_description__ = 'MindSpeed-LLM for LLMs of Ascend'
__keywords__ = 'Ascend, langauge, deep learning, NLP'
__package_name__ = 'mindspeed_llm'
__contact_names__ = 'Ascend'


with open("README.md", "r") as fh:
    long_description = fh.read()

###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")
cmd_class = {}
exts = []
try:
    import imp
    import os
    import glob
    from torch_npu.utils import cpp_extension
    from torch.utils.cpp_extension import BuildExtension

    ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME")
    op_files = []
    op_files += glob.glob("mindspeed_llm/te/ops/csrc/*.cpp")
    op_files += glob.glob("mindspeed_llm/te/ops/csrc/cann/*.cpp")
    ext_ops = cpp_extension.NpuExtension(
        name="ascendspeed_te_ops",
        sources=op_files,
        extra_compile_args=[
        '-Wno-sign-compare',
        '-Wno-deprecated-declarations',
        '-Wno-return-type',
        '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
        '-I' + imp.find_module('torch_npu')[1] + "/include/third_party/acl/inc",
        '-I' + ASCEND_TOOLKIT_HOME + '/include/',
        '-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack',
        '-fPIC', '-pie', '-Wl,--disable-new-dtags,--rpath', '-s',
    ],
    )
    exts.append(ext_ops)
    cmd_class = {"build_ext": BuildExtension.with_options(use_ninja=False)}
except Exception:
    print('Can not find env : ASCEND_TOOLKIT_HOME or ATB_HOME_PATH, ops setup failed')

setuptools.setup(
    package_data={'mindspeed_llm':['mindspeed_llm/data/Makefile']},
    name=__package_name__,
    # Versions should comply with PEP440.
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    author=__contact_names__,
    maintainer=__contact_names__,
    # The licence under which the project is released
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
    cmdclass=cmd_class,
    ext_modules=exts
)
