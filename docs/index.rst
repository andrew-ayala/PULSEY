PULSEY
======

PULSEY is a Python package for simulating stellar pulsations with
time-dependent spherical harmonic surface maps. It builds on
``jaxoplanet.starry`` to combine pulsation modes, compute disk-integrated flux
curves, render evolving stellar surfaces, and sample local surface fluxes on
longitude-latitude grids.

The package is designed around a compact workflow:

* define one or more pulsation modes with ``(l, m)`` spherical harmonic indices,
* assign frequencies, amplitudes, and phases to those modes,
* evaluate the evolving map or light curve over time,
* inspect the surface at specific map coordinates with
  :meth:`PULSEY.star.discretizeSurface`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
