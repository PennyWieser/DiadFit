========================
Installation & Updating
========================

Installation
============

First, obtain Python3 (tested on V3.9). If you haven't used python before, we recomend installing it through anaconda.
 `anaconda3 <https://www.anaconda.com/products/individual>`_.

All of the You Tube examples shown here use Jupyter Lab, which is an easy-friendly editor that will be installed along with Anaconda.

DiadFit can be installed using pip in one line. If you are using a terminal, enter:

.. code-block:: python

   pip install DiadFit

If you are using Jupyter Notebooks or Jupyter Lab, you can also install it by entering the following code into a notebook cell (note the !):

.. code-block:: python

   !pip install DiadFit

You then need to import DiadFit into the script you are running code in. In all the examples, we import Themobar as pt.:

.. code-block:: python

   import DiadFit as pf

This means any time you want to call a function from DiadFit, you do pt.function_name.



Updating
========

To upgrade to the most recent version of DiadFit, type the following into terminal:

.. code-block:: python

   pip install DiadFit --upgrade

Or in your Jupyter environment:

.. code-block:: python

   !pip install DiadFit --upgrade


For maximum reproducability, you should state which version of DiadFit you are using. If you have imported DiadFit as pt, you can find this using:

.. code-block:: python

    pf.__version__