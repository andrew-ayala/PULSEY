Installation
============

Install PULSEY from a local checkout while developing:

.. code-block:: bash

   pip install -e .

Runtime dependencies are declared in ``pyproject.toml``. The current package
uses ``numpy``, ``jax``, and ``jaxoplanet`` for numerical work and STARRY map
evaluation.

Building Documentation Locally
------------------------------

Install the documentation dependency and build the HTML pages:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

The Read the Docs configuration in ``.readthedocs.yaml`` uses the same Sphinx
configuration file.
