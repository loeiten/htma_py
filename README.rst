HTMA_py
=======

 |python| |test| |codecov| |lint| |license| |code_style| |mypy| |bandit|

.. |python| image:: https://img.shields.io/badge/python->=3.6-blue.svg
    :alt: supported python versions
    :target: https://www.python.org/

.. |test| image:: https://github.com/loeiten/htma_py/workflows/Test/badge.svg?branch=main
    :alt: test status
    :target: https://github.com/loeiten/htma_py/actions?query=workflow%3A%22Test%22

.. |codecov| image:: https://codecov.io/gh/loeiten/htma_py/branch/main/graph/badge.svg
    :alt: codecov percentage
    :target: https://codecov.io/gh/FIXME/FIXME

.. |lint| image:: https://github.com/loeiten/htma_py/workflows/Lint/badge.svg?branch=main
    :alt: lint status
    :target: https://github.com/loeiten/htma_py/actions?query=workflow%3A%22Lint%22

.. |license| image:: https://img.shields.io/badge/license-LGPL--3.0-blue.svg
    :alt: licence
    :target: https://github.com/loeiten/htma_py/blob/main/LICENSE

.. |code_style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: code standard
    :target: https://github.com/psf/black

.. |mypy| image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :alt: checked with mypy
    :target: http://mypy-lang.org/

.. |bandit| image:: https://img.shields.io/badge/security-bandit-yellow.svg
    :alt: security status
    :target: https://github.com/PyCQA/bandit

Overview
--------

Python implementation of ideas covered in `How to measure anything <https://www.howtomeasureanything.com>`_.


Getting Started
---------------

Scripts covering a variety of topics can be found in the |scripts|_ directory, and can be executed from this directory by running

.. code:: sh

   python -m scripts.name_of_script

The package implementation can be found in the |htma_py|_ directory.

Any plots from the scripts will be stored in a directory called ``plots/`` located at the root of this repo.

.. |scripts| replace:: ``scripts/``
.. _scripts: https://github.com/loeiten/htma_py/tree/main/scripts

.. |htma_py| replace:: ``htma_py/``
.. _htma_py: https://github.com/loeiten/htma_py/tree/main/htma_py

Prerequisites
-------------

- ``numpy>=1.20.2``
- ``scipy>=1.6.3``

Installing
----------

The package can be installed from source

.. code:: sh

   python setup.py install
