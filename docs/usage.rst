Usage
=====

Create A Pulsating Star
-----------------------

.. code-block:: python

   import numpy as np
   import PULSEY as p

   time = np.linspace(0.0, 1.0, 200)

   pulse = p.star(
       lmModes=[[1, 0], [2, 1]],
       freq=[1.0, 2.5],
       amp=[0.01, 0.005],
       phase=[0.0, 0.25],
       inc=90.0,
   )

   flux = pulse.computeFlux(time)

``computeFlux`` returns the disk-integrated flux at each requested time.

Compute Map Coefficients
------------------------

.. code-block:: python

   coeffs = pulse.computeMap(time)

Each row in ``coeffs`` is the dense spherical harmonic coefficient vector used
by ``jaxoplanet.starry`` at one time step.

Sample Surface Fluxes
---------------------

Use :meth:`PULSEY.star.discretizeSurface` to evaluate the local map intensity at
specific longitude and latitude points. This is surface sampling, not a
disk-integrated light curve.

Generate a full latitude-longitude grid:

.. code-block:: python

   lon, lat, surface_flux = pulse.discretizeSurface(
       time=0.0,
       nLon=72,
       nLat=36,
   )

Evaluate paired coordinates:

.. code-block:: python

   lon, lat, surface_flux = pulse.discretizeSurface(
       time=[0.0, 0.25],
       lon=[0.0, 90.0],
       lat=[0.0, 45.0],
       grid=False,
   )

When multiple times are supplied, the returned flux array prepends a time axis
to the coordinate shape.

Animate A Surface
-----------------

.. code-block:: python

   pulse.Animate(time)

The animation is saved as ``Pulsation.gif`` and displayed inline when running in
an IPython or notebook environment.
