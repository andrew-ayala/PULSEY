"""Tools for simulating non-radial stellar pulsations with STARRY maps.

PULSEY represents a pulsating stellar photosphere as a time-dependent sum of
real spherical harmonic coefficients. Each pulsation mode is described by an
``(l, m)`` pair, a frequency, an amplitude, and an initial phase. The class
below translates those user-facing mode properties into the coefficient vector
expected by :mod:`jaxoplanet.starry`, then evaluates either disk-integrated
light curves or local surface intensities.
"""

### Package imports ###
import sys
import os
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from jax import numpy as jnp
import jaxoplanet as jx
import jaxoplanet.starry as starry
from jaxoplanet.starry.light_curves import surface_light_curve
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation
from IPython.display import HTML, display
from tqdm import tqdm
from PIL import Image
import warnings

# ``s2fft`` is useful for future spherical-harmonic transforms, but PULSEY's
# current public methods do not require it. Keep it optional so users without
# the compiled backend can still import the package and compute pulsations.
try:
    import s2fft
except ImportError:
    s2fft = None


def _as_mode_array(values, n_modes, name):
    """Return a one-dimensional float array with one value per pulsation mode.

    The public API allows users to pass scalars for one-mode stars. Internally,
    PULSEY always uses length-``n_modes`` arrays so vectorized coefficient
    calculations can treat every mode in the same way.
    """
    if n_modes == 0:
        return np.array([], dtype=float)

    array = np.atleast_1d(np.asarray(values, dtype=float))

    # A scalar value is convenient for the common one-mode case and can also be
    # broadcast over multiple modes when the same property should apply to all.
    if array.size == 1 and n_modes > 1:
        array = np.full(n_modes, array.item(), dtype=float)

    if array.size != n_modes:
        raise ValueError(
            f"{name} must contain either one value or one value per lmModes entry "
            f"({n_modes} expected, {array.size} received)."
        )

    return array


# Star class object to construct stellar pulsation.
class star:
    """A pulsating stellar surface represented by spherical harmonics.

    Parameters
    ----------
    lmModes : array_like of int, shape ``(n_modes, 2)``
        Degree/order pairs ``[l, m]`` for each pulsation mode.
    freq : float or array_like of float
        Pulsation frequencies corresponding to ``lmModes``.
    amp : float or array_like of float
        Pulsation amplitudes corresponding to ``lmModes``.
    phase : float or array_like of float
        Initial phases, expressed in cycles from 0 to 1.
    inc : float, default=90
        Inclination angle in degrees.
    obl : float, default=0.0
        Obliquity angle in degrees.
    lMax : int, optional
        Maximum spherical harmonic degree to include. Defaults to the largest
        ``l`` value in ``lmModes``.
    fcn : callable, optional
        Optional transform applied to sampled surface intensities.
    osParam : int, default=2
        Stored oversampling parameter for transform functions.
    observed : bool, default=True
        If true, mode amplitudes are interpreted as observed disk-integrated
        amplitudes and are rescaled onto the intrinsic map coefficients.

    Notes
    -----
    The class name is kept lowercase for compatibility with existing notebooks.
    New code can still instantiate it with ``p.star(...)`` after importing
    ``PULSEY as p``.
    """

    # Initialization function accepting the mode list and mode properties.
    def __init__(
        self,
        lmModes,
        freq,
        amp,
        phase,
        inc=90,
        obl=0.0,
        lMax=None,
        fcn=None,
        osParam=2,
        observed=True,
    ):
        # Normalize mode inputs up front so the rest of the code can assume a
        # two-column integer array. A single ``[l, m]`` pair is accepted.
        self.lmModes = np.asarray(lmModes, dtype=int)
        if self.lmModes.ndim == 1:
            if self.lmModes.size == 0:
                self.lmModes = self.lmModes.reshape(0, 2)
            elif self.lmModes.size == 2:
                self.lmModes = self.lmModes.reshape(1, 2)
            else:
                raise ValueError("lmModes must be empty, a single [l, m] pair, or an array of [l, m] pairs.")

        if self.lmModes.ndim != 2 or self.lmModes.shape[1] != 2:
            raise ValueError("lmModes must have shape (n_modes, 2).")

        # Store one frequency, amplitude, and phase per requested mode.
        n_modes = len(self.lmModes)
        self.freq = _as_mode_array(freq, n_modes, "freq")
        self.amp = _as_mode_array(amp, n_modes, "amp")
        self.phase = _as_mode_array(phase, n_modes, "phase")

        # Orientation is stored in degrees for user-facing attributes. The
        # underlying STARRY surface receives radians below.
        self.inc = inc
        self.obl = obl

        # Optional transform function and metadata. The transform is applied by
        # ``discretizeSurface`` to local surface intensity samples.
        self.fcn = fcn
        self.osParam = osParam

        self.observed = observed
        self.binaryFlag = False
        self.nSignals = np.arange(n_modes)
        self.unphysical = False

        # Determine the maximum spherical harmonic degree represented in the
        # coefficient vector. The dense STARRY vector has indices up to lMax.
        if lMax is None:
            self.lMax = int(np.max(self.lmModes[:, 0])) if n_modes > 0 else 1
        else:
            self.lMax = int(lMax)

        if n_modes > 0 and self.lMax < int(np.max(self.lmModes[:, 0])):
            raise ValueError("lMax must be at least as large as the largest l value in lmModes.")

        # Build a complete Ylm coefficient dictionary through lMax. The radial
        # (0, 0) term is initialized to 1 so the unperturbed surface has unit
        # disk-integrated flux.
        y_data = {(0, 0): 1.0}
        for ell in range(1, self.lMax + 1):
            for order in range(-ell, ell + 1):
                y_data[(ell, order)] = 0.0
        y = starry.ylm.Ylm(data=y_data)

        # STARRY uses radians. The historical PULSEY convention offsets the
        # inclination by 90 degrees before giving it to the surface map.
        self._map = starry.surface.Surface(
            y=y,
            inc=(self.inc + 90.0) * np.pi / 180.0,
            obl=self.obl * np.pi / 180.0,
        )

        # Keep a SurfaceSystem available even for isolated stars so binary
        # insertion can update the same object graph.
        self.system = starry.orbit.SurfaceSystem(
            central=jx.orbits.keplerian.Central(mass=1.0, radius=1.0),
            central_surface=self._map,
        )

        # Calibrate user amplitudes/phases and initialize the map at time zero.
        self._pulsationCorrections()
        initCoeffs = self._singleMap(0.0)
        self._map.y.data.update(starry.ylm.Ylm.from_dense(initCoeffs, normalize=False).data)

        if self.fcn is not None:
            self.setTransFcn(self.fcn, osParam)

    def setTransFcn(self, fcn, osParam=2):
        """Store a surface-intensity transform function.

        Parameters
        ----------
        fcn : callable
            Function applied to local surface intensity samples returned by
            :meth:`discretizeSurface`.
        osParam : int, default=2
            Oversampling metadata retained for compatibility with older PULSEY
            notebooks that passed this value during initialization.
        """
        self.fcn = fcn
        self.osParam = osParam

    # Calibration of spherical harmonic mode amplitude and phase coefficients
    # to desired output values. Pulsation simulation of non-axisymmetric modes
    # is done by combining positive and negative m components at offset phases.
    def _pulsationCorrections(self):
        """Compute per-mode amplitude scale factors and phase offsets."""
        self.ampScaleFactor = np.ones(len(self.nSignals))
        self._phaseOffsetArray = np.zeros(len(self.nSignals))

        # Nothing needs calibration when the user requests a static surface.
        if len(self.nSignals) == 0:
            return

        # Evaluate each requested mode at quarter-phase intervals. These samples
        # let PULSEY infer how an intrinsic coefficient maps onto the observed
        # disk-integrated amplitude at the current inclination.
        for i in range(len(self.lmModes)):
            timeSample = np.arange(0.0, 1.0, 0.25)
            testFluxArray = np.zeros(len(timeSample))

            ell = self.lmModes[i][0]
            order = self.lmModes[i][1]

            # Reset the map to the unperturbed state before measuring the
            # response of the single mode under consideration.
            for key in self._map.y.data:
                self._map.y.data[key] = 0.0
            self._map.y.data.update({(0, 0): 1.0})

            for j, t in enumerate(timeSample):
                posC, negC = _LxMx(t, order, frequency=1.0, amp=1.0, phase0=0.0)
                self._map.y.data.update({(ell, np.abs(order)): posC})
                if order != 0:
                    self._map.y.data.update({(ell, -np.abs(order)): negC})

                # Store the disk-integrated flux for this pure mode sample.
                flux = surface_light_curve(self._map)
                testFluxArray[j] = np.asarray(flux)

            # If amplitudes are specified as observed light-curve amplitudes,
            # convert them back into the intrinsic coefficient scale required by
            # the surface map. Pole-on or cancellation-heavy modes may have a
            # near-zero response, so keep their intrinsic scale unchanged.
            if self.observed:
                maxAmp = np.nanmax(testFluxArray) - 1.0
                if np.isclose(maxAmp, 0.0):
                    warnings.warn(
                        f"Mode ({ell}, {order}) has near-zero observed amplitude at this orientation; "
                        "leaving its intrinsic amplitude scale unchanged."
                    )
                    self.ampScaleFactor[i] = 1.0
                else:
                    self.ampScaleFactor[i] = 1.0 / maxAmp

            # Shift the user phase so phase zero starts at the average flux
            # point immediately before the maximum.
            if timeSample[np.argmax(testFluxArray)] != 0:
                self._phaseOffsetArray[i] = timeSample[np.argmax(testFluxArray)] - 0.25
            else:
                self._phaseOffsetArray[i] = 0.75

    # Computation of spherical harmonic coefficient values at a single time.
    # This helper is vectorized by ``computeMap`` for arrays of times.
    def _singleMap(self, time):
        """Return the dense Ylm coefficient vector at a single time."""
        coeffArray = jnp.zeros(lmIndex(self.lMax, self.lMax) + 1)
        coeffArray = coeffArray.at[0].set(1.0)

        if len(self.lmModes) == 0:
            return coeffArray

        ell = self.lmModes[:, 0]
        order = self.lmModes[:, 1]

        # Convert user mode properties into cosine/sine coefficient pairs. The
        # positive-m term stores the cosine component; the negative-m term
        # stores the sine component that sets the direction of the standing wave.
        posC, negC = _LxMx(
            time,
            order,
            frequency=self.freq[:],
            amp=self.amp[:] * self.ampScaleFactor[:],
            phase0=self.phase[:] + self._phaseOffsetArray[:],
        )

        # Add all mode contributions into the dense STARRY coefficient vector.
        # Repeated modes are intentionally summed.
        coeffArray = coeffArray.at[lmIndex(ell[:], np.abs(order[:]))].add(posC[:])
        coeffArray = coeffArray.at[lmIndex(ell[:], -np.abs(order[:]))].add(negC[:])

        return coeffArray

    def _surfaceFromCoeffs(self, coeffArray):
        """Build an isolated STARRY surface from a coefficient vector.

        This avoids mutating ``self._map`` when sampling local intensities, which
        is useful for evaluating many times or coordinates in a row.
        """
        return starry.surface.Surface(
            y=starry.ylm.Ylm.from_dense(coeffArray, normalize=False),
            inc=self._map._inc,
            obl=self._map._obl,
            u=self._map.u,
            period=self._map.period,
            amplitude=self._map.amplitude,
            normalize=False,
            phase=self._map.phase,
            radius=self._map.radius,
            shear=self._map.shear,
        )

    # JAX vmap function to iterate _singleMap over an array of time values.
    def computeMap(self, timeArray):
        """Compute surface-map coefficients for an array of time values.

        Parameters
        ----------
        timeArray : array_like of float
            Time values at which to evaluate the pulsation state.

        Returns
        -------
        array_like
            Two-dimensional array whose rows are dense Ylm coefficient vectors.
        """
        coeffArray = jax.vmap(self._singleMap)(timeArray)
        return coeffArray

    # Retrieve the disk-integrated flux from the surface map at a single time.
    # This helper is vectorized by ``computeFlux`` for arrays of times.
    def _singleFlux(self, time):
        """Return the disk-integrated flux at a single time."""
        if self.binaryFlag:
            return self.binaryFlux(time)

        coeffArray = self._singleMap(time)
        testMap = self._surfaceFromCoeffs(coeffArray)
        flux = surface_light_curve(testMap)
        return flux

    # JAX vmap function to iterate _singleFlux over an array of time values.
    def computeFlux(self, timeArray):
        """Compute disk-integrated fluxes for an array of time values.

        Parameters
        ----------
        timeArray : array_like of float
            Time values at which to evaluate the light curve.

        Returns
        -------
        array_like
            Disk-integrated flux at each time step.
        """
        flux = jax.vmap(self._singleFlux)(timeArray)
        return flux

    # Initialize star into a binary system with default orbital parameters.
    def insertBinary(self, m1=1.0, r1=1.0, m2=1.0, r2=1.0, period=1.0, tTransit=0.0):
        """Insert the pulsating star as the primary of an eclipsing binary.

        Parameters
        ----------
        m1 : float, default=1.0
            Mass of the primary star.
        r1 : float, default=1.0
            Radius of the primary star.
        m2 : float, default=1.0
            Mass of the secondary star.
        r2 : float, default=1.0
            Radius of the secondary star.
        period : float, default=1.0
            Orbital period of the binary system.
        tTransit : float, default=0.0
            Time of eclipse occultation transit.
        """
        central = jx.orbits.keplerian.Central(mass=m1, radius=r1)
        secondary = jx.orbits.keplerian.Body(
            time_transit=tTransit,
            period=period,
            mass=m2,
            radius=r2,
        )
        self.system = starry.orbit.SurfaceSystem(central=central, central_surface=self._map)
        self.system = self.system.add_body(secondary)
        self.binaryFlag = True

        return "Star inserted into binary system."

    # Calculate the coefficients for the primary star in a binary system at a
    # given time value. This is vectorized by ``computeBinary``.
    def binaryFlux(self, time=0.0):
        """Compute flux output from the pulsating star in an eclipsing binary.

        Parameters
        ----------
        time : float, default=0.0
            Time value at which to evaluate the binary light curve.

        Returns
        -------
        float
            Binary-system flux at ``time``.
        """
        newCoeffs = self._singleMap(time)
        self.system.central_surface.y.data.update(starry.ylm.Ylm.from_dense(newCoeffs, normalize=False).data)
        flux = jx.starry.light_curves.light_curve(self.system)
        return flux(time)[0]

    # JAX vmap function to iterate binaryFlux over an array of time values.
    def computeBinary(self, timeArray):
        """Compute binary-system fluxes for an array of time values.

        Parameters
        ----------
        timeArray : array_like of float
            Time values at which to evaluate the binary light curve.

        Returns
        -------
        array_like
            Binary-system flux at each time step.
        """
        flux = jax.vmap(self.binaryFlux)(timeArray)
        return flux

    def discretizeSurface(
        self,
        time=0.0,
        lon=None,
        lat=None,
        nLon=72,
        nLat=36,
        degrees=True,
        grid=True,
    ):
        """Sample local surface fluxes on longitude and latitude coordinates.

        Parameters
        ----------
        time : float or array_like of float, default=0.0
            Time or times at which to evaluate the pulsating surface.
        lon : float or array_like of float, optional
            Longitude coordinate(s). If omitted, an evenly spaced longitude grid
            is generated using ``nLon`` samples.
        lat : float or array_like of float, optional
            Latitude coordinate(s). If omitted, an evenly spaced latitude grid is
            generated using ``nLat`` samples.
        nLon : int, default=72
            Number of generated longitude samples when ``lon`` is omitted.
        nLat : int, default=36
            Number of generated latitude samples when ``lat`` is omitted.
        degrees : bool, default=True
            Interpret and return coordinates in degrees. If false, radians are
            used throughout.
        grid : bool, default=True
            If true, make a full latitude-longitude mesh. If false, evaluate
            paired/broadcast coordinates point-by-point.

        Returns
        -------
        lonPoints : array_like
            Longitude coordinates with the same coordinate shape as ``flux``.
        latPoints : array_like
            Latitude coordinates with the same coordinate shape as ``flux``.
        flux : array_like
            Local surface intensity, or specific surface flux, at each point.
            For multiple times, the time axis is prepended to the coordinate
            shape.

        Notes
        -----
        This method samples the map itself. It is not disk-integrated; use
        :meth:`computeFlux` for observed light curves.
        """
        if lon is None:
            if int(nLon) < 1:
                raise ValueError("nLon must be at least 1.")
            if degrees:
                # Drop the duplicated endpoint so the grid does not sample both
                # -180 and +180 degrees.
                lon_values = jnp.linspace(-180.0, 180.0, int(nLon) + 1)[:-1]
            else:
                lon_values = jnp.linspace(-jnp.pi, jnp.pi, int(nLon) + 1)[:-1]
        else:
            lon_values = jnp.asarray(lon)

        if lat is None:
            if int(nLat) < 1:
                raise ValueError("nLat must be at least 1.")
            if degrees:
                lat_values = jnp.linspace(-90.0, 90.0, int(nLat))
            else:
                lat_values = jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, int(nLat))
        else:
            lat_values = jnp.asarray(lat)

        # Build either a full mesh or a point-wise/broadcast coordinate array.
        if grid:
            lonPoints, latPoints = jnp.meshgrid(lon_values, lat_values)
        else:
            lonPoints, latPoints = jnp.broadcast_arrays(lon_values, lat_values)

        if degrees:
            lonRad = lonPoints * jnp.pi / 180.0
            latRad = latPoints * jnp.pi / 180.0
        else:
            lonRad = lonPoints
            latRad = latPoints

        # ``Surface.intensity`` evaluates local map brightness at rest-frame
        # coordinates. Looping over time avoids mutating the stored map and keeps
        # arbitrary user transform functions usable.
        scalar_time = np.ndim(time) == 0
        time_values = np.atleast_1d(np.asarray(time, dtype=float))
        fluxes = []
        for t in time_values:
            coeffArray = self._singleMap(float(t))
            sampleMap = self._surfaceFromCoeffs(coeffArray)
            flux = sampleMap.intensity(latRad, lonRad)
            if self.fcn is not None:
                flux = self.fcn(flux)
            fluxes.append(flux)

        flux = jnp.stack(fluxes)
        if scalar_time:
            flux = flux[0]

        return lonPoints, latPoints, flux

    # Show visual representation of star object.
    def show(self, time=None, inc=None, obl=None, phase=0.0, cmap="seismic_r", **kwargs):
        """Display the pulsating stellar surface at a selected time.

        Parameters
        ----------
        time : float, optional
            Time at which to render the pulsation state. If omitted, the current
            map coefficients are rendered.
        inc : float, optional
            New inclination angle in degrees for the display.
        obl : float, optional
            New obliquity angle in degrees for the display.
        phase : float, default=0.0
            Reserved for compatibility with earlier notebook examples.
        cmap : str, default="seismic_r"
            Matplotlib colormap used for the rendered surface.
        **kwargs
            Additional keyword arguments reserved for future plotting options.
        """
        fig1, ax1 = plt.subplots(figsize=(4.25, 4.25))

        if inc is not None:
            self.inc = inc
            self._map.inc = (self.inc + 90.0) * np.pi / 180.0
        if obl is not None:
            self.obl = obl
            self._map.obl = self.obl * np.pi / 180.0

        if time is not None:
            coeffArray = self._singleMap(time)
            self._map.y.data.update(starry.ylm.Ylm.from_dense(coeffArray, normalize=False).data)

        starry.visualization.show_surface(self._map, cmap=cmap, ax=ax1)

    # Plot graph relating two parameters of star object.
    def plot(self, var1, var2):
        """Plot one PULSEY quantity against another.

        Parameters
        ----------
        var1 : array_like
            Values for the x-axis, commonly time.
        var2 : array_like
            Values for the y-axis, commonly flux.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(var1, var2, lw=2, alpha=1)
        plt.scatter(var1, var2, color="black", alpha=0.25, s=10)
        plt.xlabel("Time [s]", fontsize=20)
        plt.ylabel("Flux [normalized]", fontsize=20)
        plt.show()

    # Create animation to visualize pulsation flux variabilities of star.
    def Animate(self, timeArray):
        """Animate the pulsating stellar surface over a time array.

        Parameters
        ----------
        timeArray : array_like of float
            Time values to render into the animation frames.
        """
        # Render the surface values first so the color scale can be chosen from
        # the full sequence.
        coeffArray = self.computeMap(timeArray)

        def render(coeffs):
            self._map.y.data.update(starry.ylm.Ylm.from_dense(coeffs, normalize=False).data)
            return self._map.render()

        rendered = jax.vmap(render)(coeffArray)

        # Find the most extreme surface brightness excursions for a symmetric
        # color range around the uniform-map value.
        vRange = np.nanmax(np.abs(np.array(rendered).flatten() - 1 / np.pi))
        vmid = 1.0 / np.pi

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        im = ax.imshow(
            rendered[0],
            cmap="seismic_r",
            animated=True,
            vmin=vmid - vRange,
            vmax=vmid + vRange,
            origin="lower",
        )

        def update(frame):
            im.set_data(rendered[frame])
            return [im]

        if len(rendered) > 1500:
            warnings.warn("WARNING: Animation too long. Will limit animation")
            rendered = rendered[:1500]

        anim = animation.FuncAnimation(fig, func=update, frames=len(rendered), interval=50, blit=True)
        writergif = animation.PillowWriter(fps=30)
        anim.save("Pulsation.gif", writer=writergif)
        display(HTML(anim.to_jshtml()))
        plt.close(fig)


# Function to determine position of a spherical harmonic mode coefficient in
# the dense STARRY Ylm coefficient array.
def lmIndex(l, m):
    """Return the dense STARRY coefficient index for spherical harmonic ``(l, m)``.

    STARRY stores real spherical harmonic coefficients in a one-dimensional
    vector using ``index = l**2 + l + m``.
    """
    return (l**2 + l) + m


# Function to calculate coefficients used to construct a standing-wave
# pulsation by combining positive and negative m modes.
def _LxMx(t, m, frequency=1, amp=1, phase0=0):
    """Construct pulsation coefficients for a spherical harmonic mode.

    Parameters
    ----------
    t : float or array_like
        Time value(s) of the periodic pulsation.
    m : int or array_like of int
        Azimuthal order(s). The sign determines the sine component direction.
    frequency : float or array_like of float, default=1
        Pulsation frequency.
    amp : float or array_like of float, default=1
        Pulsation amplitude.
    phase0 : float or array_like of float, default=0
        Phase offset in cycles.

    Returns
    -------
    posCoeff : float or array_like
        Cosine coefficient for the positive ``m`` component.
    negCoeff : float or array_like
        Sine coefficient for the negative ``m`` component.
    """
    posCoeff = amp * jnp.cos(2 * jnp.pi * ((frequency * t) + phase0))
    negCoeff = -jnp.sign(m) * amp * jnp.sin(2 * jnp.pi * ((frequency * t) + phase0))
    return posCoeff, negCoeff
