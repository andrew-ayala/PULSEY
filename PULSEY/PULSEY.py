### Description:
"""
This package creates simulations of periodic non-radial stellar pulsation.  These pulsations are either driven waves that propogate across the surface of a 
star due to pressure and/or gravitational gradients, known as p-mode and g-mode waves respectively. These pulsation waves can be modeled by oscillating
spherical harmonics.  Using the "STARRY" python package one can model differential magnitudes of a spherical surface. However, there 
are no methods within this package enabling these surfaces to evolve or vary over time-- or in another word, pulsate. This program
utilizes these static maps to create a periodically pulsating stellar source by summing various spherical harmonic magnitude values
over time.  
"""

### Package Imports ###
import sys
import os
import numpy as np
import jax
from jax import numpy as jnp
import jaxoplanet as jx
import jaxoplanet.starry as starry
from jaxoplanet.starry.light_curves import surface_light_curve
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from PIL import Image
import warnings

# Star class object to construct stellar pulsation
class star:
    
    """
    ### Description

    Initialize star object to simulate stellar pulsation

    ### Parameters

    lmModes : array_like (int, 2D)
        Array of L-M modes of stellar pulsation
    freq : array_like (float)
        Pulsation frequencies of respective L-M modes in lmModes
    amp : array_like (float)
        Pulsation amplitudes of respective L-M modes in lmModes
    phase : array_like (float)
        Initial relative phase of L-M mode pulsations in lmModes
    inc : float (default = 90.0)
        Inclination angle (degrees) of star object relative to observer
    obl : float (default = 90.0)
        Obliquity angle (degrees) of star object relative to observer
    lMax : int (default = None)
        Maximum value L-mode complexity of star object
    fcn : float function (default = None)
        Set user function for transforming surface map values star object
    osParam : int (default = 2)
        Integer to determine resolution of pixels produced from surface map transform
    observed : bool (default = True)
        Boolean flag determining whether amplitudes are associated with observed values or intrinsic values
    
    ### Returns

    star : tuple
        Star object with pulsation modes, frequencies, and amplitudes as indicated by user

    ### Example
    
    """

    # Initialization function accepting arguments of freq, amp, etc. to initialize star with features
    def __init__(self, lmModes, freq, amp, phase, inc=90, obl = 0.0, lMax = None, fcn = None, osParam = 2, observed = True):
        
        # Initialize all self attributes to input/default values
        self.lmModes = np.array(lmModes)
        self.freq = np.array(freq)
        self.amp = np.array(amp)
        self.phase = np.array(phase) # from 0 to 1
        self.inc = inc # in degrees
        self.obl = obl # in degress
        self.fcn = fcn
        self.observed = observed
        self.binaryFlag = False
        self.nSignals = np.arange(len(lmModes))
        self.unphysical = False
        
        
        # Determine the maximum L-value compolexity of star
        if lMax is None:
            if(len(lmModes) > 0):
                self.lMax = np.max(self.lmModes)
            else:
                self.lMax = 1
        else:
            self.lMax = lMax

        ### TESTING: [0,0] mode
        # if [0,0] in lmModes:
        #     l = 0
        # else:
        #     l = 1

        # Construct Ylm data dictionary and initialize all amplitude values to 0 (except for the [0,0] radial mode which is set to 1)
        y = starry.ylm.Ylm(data = {tuple((0,0)): 1.0})
        l = 1
        # Loop to populate Ylm data values
        while l <= self.lMax: 
            m = -l
            while m <= l:
                newMode = {tuple((l,m)): 0.0}
                y.data.update(newMode)
                m += 1
            l += 1

        # Initialize star with constructed Ylm surface map and add binary system attribute initialized to default values
        self._map = starry.surface.Surface(y = y, inc=(self.inc+90) * np.pi/180.0)
        self.system = starry.orbit.SurfaceSystem(central=jx.orbits.keplerian.Central(mass=1.0, radius=1.0), central_surface=self._map)

        # Calculation function to recalibrate the Ylm data to appropriate amplitudes and phases as indicated by user
        self._pulsationCorrections()

        # Determine whether to apply a transform function to output surface map flux values as indicated by user
        if self.fcn is not None:
            self.setTransFcn(self.fcn, osParam)
            self.lat, self.lon, self.Y2P, self.P2Y, self.Dx, self.Dy = self._map.get_pixel_transforms(oversample=osParam)


    # Calibration of spherical harmonic (SH) mode amplitude and phase coefficents to desired output values.
    # Pulsation simulation of individual modes is done by conglomerating positive and negative m≠0 modes at offset phases
    def _pulsationCorrections(self):
        
        # Initialization of arrays to contain coefficients for amplitude and phase calibration
        self.ampScaleFactor = np.ones(len(self.nSignals))
        self._phaseOffsetArray = np.zeros(len(self.nSignals))

        # Loop over all user input modes to calibrate proper phases and amplitudes for pulsation
        for i in range(len(self.lmModes)):
            
            # Initialize time array to sample every quarter phase of pulsation (max flux, min flux, and two avg flux points) and flux array to record output values at these phases
            timeSample = np.arange(0.0,1.0,0.25) # Time array with four points separated by a quarter phase
            testFluxArray = np.zeros(len(timeSample))

            # Grab l and m values of current pulsation mode iteration
            l = self.lmModes[i][0]
            m = self.lmModes[i][1]

            # Default each SH-mode amplitude of initialized star to 0, except for the radial [0,0] mode amplitude which is set to 1
            for key in self._map.y.data:
                self._map.y.data[key] = 0.0
            self._map.y.data.update({(0,0) : 1.0})

            # Call helper function _LxMx to calculate the necessary amplitude coefficents for a single SH-mode
            # In the case of m≠0, posC coefficient is applied to the positive m-value and negC coefficient is applied the negative m-value at a given l-value
            # Otherwise, only the posC coefficient is applied to the m=0 mode at a given l
            for j, t in enumerate(timeSample):
                posC,negC = _LxMx(t, m, frequency=1.0, amp=1.0, phase0=0.0)
                self._map.y.data.update({(l,np.abs(m)) : posC})
                if m != 0:
                    self._map.y.data.update({(l,-np.abs(m)) : negC})

                # Store the output flux of the given mode
                flux = starry.light_curves.surface_light_curve(self._map)
                testFluxArray[j] = flux

            # Calibrate amplitude coefficients to observed values if flag is true, otherwise keep intrinsic amplitude values for SH-mode
            if self.observed:
                maxAmp = np.nanmax(testFluxArray) - 1.0
                self.ampScaleFactor[i] = 1.0/maxAmp
                ### TESTING: unphysical amplitude user warning
                # if(self.ampScaleFactor[i] > 1.0):
                #     warnings.warn("WARNING: Producing unphysical amplitude values!")
                #     self.unphysical = True

            # Default and save the phase value to average flux value before increase in amplitude
            if timeSample[np.argmax(testFluxArray)] != 0:
                self._phaseOffsetArray[i] = (timeSample[np.argmax(testFluxArray)]-0.25)
            else:
                self._phaseOffsetArray[i] = 0.75
                
            
    #Transform constructed stellar surface map to new values in accordance with input transform function
    def setTransFcn(self, fcn, osParam=2):
        """
        ### Description
        
        Set function for transforming surface map pixel magnitudes

        ### Parameters

        fcn : float (default = 1.0)
            Equation to transform surface magnitudes to observable values

        osParam: int (default = 2)
            Degree to which surface map of star will be granulated to pixels. Higher value equals more pixels.

        ### Returns

        ### Example
        
        """
        self.fcn = fcn
        
    # Computation of SH-mode amplitude value according to mode frequency. Outputs pulsation flux output at a single input time (to be used iteratively over a JAX vmap function)
    def _singleMap(self, time):
        
        # Initialize array to contain amplitude coefficients for each SH-mode
        coeffArray = jnp.zeros(lmIndex(self.lMax, self.lMax)+1)
        coeffArray = coeffArray.at[0].set(1.0)
        l = self.lmModes[:, 0]
        m = self.lmModes[:, 1]

        # Calculate necessary amplitude and phase coefficients for each SH-mode at given time value
        posC,negC = _LxMx(time, m, frequency=self.freq[:],amp=self.amp[:]*self.ampScaleFactor[:], 
                            phase0=self.phase[:]+self._phaseOffsetArray[:])
        P = posC
        N = negC

        # Apply coefficients to their respective value postions in the array to be administered to the final surface map of star
        for i in m:
            coeffArray = coeffArray.at[lmIndex(l[:], np.abs(m[:]))].add(P[:])
            if i != 0:
                coeffArray = coeffArray.at[lmIndex(l[:],-np.abs(m[:]))].add(N[:])

        ### TESTING: apply function to coefficient values
        # if self.fcn is not None:
        #     p = self.Y2P.dot(coeffArray[:,:])
        #     newP = self.fcn(p)
        #     coeffArray[:,:] = self.P2Y.dot(newP)

        return coeffArray
    
    # JAX vmap function to iterate computePulsation() over an array of time values
    def computeMap(self, timeArray):
        """
        ### Description
        
        JAX vmap function to compute surface map SH-mode coefficients for an array of time values using the star._singleMap() function

        ### Parameters

        timeArray : array-like (float)
            Array of time values to compute surface map coefficients

        ### Returns

        coeffArray : array_like (float, 2D)
            Array of surface map SH-mode coefficients at each time step

        ### Example
        
        """
        coeffArray = jax.vmap(self._singleMap)(timeArray)
        return coeffArray
    
    # Function to retrieve output flux values from surface map after applying computed coefficients at a single time value (to be used iteratively over a JAX vmap function)
    def _singleFlux(self, time):
        
        # Determine map coefficients using computePulsation() function at a single time
        coeffArray = self._singleMap(time)

        # If in a binary system, compute the flux from the total system
        if self.binaryFlag == True:
            self.sys.primary.map.y.update(starry.Ylm.from_dense(coeffArray, normalize=False).data)
            flux = jx.starry.light_curves.light_curve(self.sys)

        # Else, if single star simply retrieve its output flux
        else:
            testMap = self._map
            testMap.y.data.update(starry.Ylm.from_dense(coeffArray, normalize=False).data)
            flux = jx.starry.light_curves.surface_light_curve(testMap)

        return flux
    
    # JAX vmap function to iterate computePulsation() over an array of time values
    def computeFlux(self, timeArray):
        """
        ### Description
        
        JAX vmap function to compute flux output from surface maps for an array of time values using the star._singleFlux() function

        ### Parameters

        timeArray : array-like (float)
            Array of time values to compute fluxes

        ### Returns

        flux : array_like (float, 2D)
            Array of star's output flux at each time step

        ### Example
        
        """
        flux = jax.vmap(self._singleFlux)(timeArray)
        return flux
    
    # Initialize star into binary system with default values and execute JAX vmap function to iterate computeBinary() over an array of time values
    def insertBinary(self, m1=1.0, r1=1.0, m2=1.0, r2=1.0, period=1.0, tTransit=0.0):
         
        """Inserts star within binary system with given paramater inputs

        ### Parameters

        m1 : float (default = 1.0)
            Mass of primary star

        r1 : float (default = 1.0)
            Radius of primary star

        m2 : float (default = 1.0)
            Mass of secondary star

        r2 : float (default = 1.0)
            Radius of secondary star

        period : float (default = 1.0)
            Orbital period of binary system

        tTransit : float (default = 0.0)
            Time of eclipse occultation transit
        
        ### Returns

        ### Example
        
        """
        central = jx.orbits.keplerian.Central(mass=m1, radius=r1)
        secondary = jx.orbits.keplerian.Body(time_transit=tTransit, period=period, mass=m2, radius=r2)
        self.system = starry.orbit.SurfaceSystem(central=central, central_surface=self._map)
        self.system = self.system.add_body(secondary)

        return "Star inserted into binary system."
    
    # Calculate the coefficients for the surface map of the primary star in a binary system at a given time value (to be used iteratively over a JAX vmap function)
    def binaryFlux(self, time=0.0):
        """
        ### Description
        
        Compute flux output from star in eclipsing binary for a single time value

        ### Parameters

        time : float (default = 0.0)
            Array of time values to compute fluxes

        ### Returns

        flux : float
            Star's output flux at a single time step

        ### Example
        
        """
        newCoeffs = self._singleMap(time)
        self.system.central_surface.y.data.update(starry.Ylm.from_dense(newCoeffs, normalize=False).data) # Applies update to actual binary system
        flux = jx.starry.light_curves.light_curve(self.system)
        return flux(time)[0]
    
    # JAX vmap function to iterate binaryFlux() over an array of time values
    def computeBinary(self, timeArray):
        """
        ### Description
        
        JAX vmap function to compute flux output from star in eclipsing binary for an array of time values using the star.binaryFlux() function

        ### Parameters

        timeArray : array-like (float)
            Array of time values to compute fluxes

        ### Returns

        flux : array_like (float, 2D)
            Array of star's output flux at each time step

        ### Example
        
        """
        flux = jax.vmap(self.binaryFlux)(timeArray)
        return flux

    # Show visual representation of star object
    def show(self, time=0.0, phase=0.0, cmap="seismic_r", **kwargs):
        """
        ### Description

        Display surface map of constructed pulsating star at given time

        ### Parameters

        time : float (default = 0.0)
            Time at which to construct the surface map

        phase : float (default = 0.0)
            Phase at which to evaluate the surface map

        cmap : string (default = 'seismic_r')
            Color map to fill in visualized star plot

        ### Example
        
        """
        fig1, ax1 = plt.subplots(figsize=(4.25, 4.25))  
        coeffArray = self._singleMap(time)
        self._map.y.data.update(starry.Ylm.from_dense(coeffArray, normalize=False).data)
        starry.visualization.show_surface(self._map, cmap = cmap, ax=ax1)

    # Plot graph relating two parameters of star object
    def plot(self,var1,var2):
        """
        ### Description

        Plot relating two parameters of star object

        ### Parameters

        var1 : array_like (float, int)
            Time at which to construct the surface map

        var2 : array_like (float, int)
            Phase at which to evaluate the surface map

        ### Example
        
        """
        plt.figure(figsize=(10, 5))
        plt.plot(var1, var2, lw=2, alpha = 1)
        plt.scatter(var1, var2, color = 'black', alpha = 0.25, s = 10)
        #plt.title("Integrated Flux over Time of Random Pulsation")
        #plt.xlim(0,1)
        plt.xlabel("Time [s]", fontsize=20)
        plt.ylabel("Flux [normalized]", fontsize=20)
        plt.show()

    # Create animation to visualize pulsation flux variablities of star
    def Animate(self, timeArray):
    
        # Render the surface values first
        coeffArray = self.computeMap(timeArray)
        
        def render(coeffs):
            self._map.y.data.update(starry.Ylm.from_dense(coeffs, normalize=False).data)
            return self._map.render()


        rendered = jax.vmap(render)(coeffArray)

        #Find most extreme surface brightness excursions for color bar
        vRange = np.nanmax(np.abs(np.array(rendered).flatten() - 1/np.pi))
        vmid = 1.0/np.pi

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.axis('off')
        im = ax.imshow(rendered[0], cmap="seismic_r", animated=True, vmin = vmid-vRange, vmax = vmid+vRange, origin = "lower")

        def update(frame):
            im.set_data(rendered[frame])
            return [im]

        anim = animation.FuncAnimation(fig, func=update, frames=len(rendered), interval=50,blit=True)
        writergif = animation.PillowWriter(fps=30)
        gif = anim.save('Pulsation.gif',writer=writergif)
        display(HTML(anim.to_jshtml()))
        plt.close(fig)

        # anim = animation.ArtistAnimation(fig, imList, interval=50, blit=True)
        # writergif = animation.PillowWriter(fps=30)
        # gif = anim.save('Pulsation.gif',writer=writergif)
        # display(HTML(anim.to_jshtml()))
        # plt.close(fig)
        #return(timeArray)

        # video = anim.to_html5_video() 
        # # embedding for the video 
        # html = display.HTML(video) 
        # # draw the animation 
        # display(html)
        # #plt.show()

# Function to determine postition of given spherical harmonic mode coefficient in Ylm array
def lmIndex(l, m):
    return (l**2 + l) + m

#Function to calculate coefficients to construct standing wave pulsation from combining +/- m-modes
def _LxMx(t, m, frequency=1,amp=1, phase0=0):
    """Construct pulsation coefficients of a single l-m spherical harmonic mode

    ### Parameters

    t : array_like (int, float)
        Array of time values of periodic pulsation
    m : int
        M-mode value to calculate sign direction of pulsation
    frequency : float (default = 1.0)
        Frequency of sinusoidal pulsation
    amp : float (deafault = 1.0)
        Amplitude of pulsation mode
    phase0 : float (default = 0.0)
        Phase offset of sinusoidal pulsation
    
    ### Returns

    posCoeff : float
        Positive coefficient of single L-M mode spherical harmonic pulsation

    negCoeff : float
        Negative coefficient of single L-M mode spherical harmonic pulsation

    ### Example
    
    """
    posCoeff = amp * jnp.cos(2*jnp.pi* ((frequency*t) +phase0))
    negCoeff = -jnp.sign(m) * amp * jnp.sin(2*jnp.pi* ((frequency*t) +phase0))
    return posCoeff, negCoeff

    #####################################################################################################################################

