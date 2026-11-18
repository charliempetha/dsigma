Installation
============

To install :code:`dsigma`, you can use pip or clone the source code from GitHub.

Automatic Installation
----------------------

The easiest way to install :code:`dsigma` is to use :code:`pip` to install the latest stable version from the Python Package Index (PyPI). This will automatically install the package and all its dependencies.

.. code-block:: none

    pip install dsigma

The following packages will be installed alongside :code:`dsigma` if not installed already.

* :code:`numpy`
* :code:`scipy`
* :code:`astropy`
* :code:`scikit-learn`
* :code:`astropy-healpix`
* :code:`tqdm`

Additionally, for calculating the lens magnification bias, the installation of the Cosmic Microwave Background (CMB) code :code:`camb` is recommended. However,  this is no hard requirement and :code:`dsigma` will run fine without it.

Manual Installation
-------------------

You can install :code:`dsigma` directly from this source repository. The
installable package is in the :code:`dsigma/` directory; similarly named files
at the repository root are not included in the package build.

The following code installs a source checkout:

.. code-block:: none

    git clone https://github.com/charliempetha/dsigma.git
    cd dsigma
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install .

For development, use an editable installation so changes under
:code:`dsigma/` are reflected immediately:

.. code-block:: none

    python -m pip install -e .
