#Decription:
"""
This package creates simulations of periodic non-radial stellar pulsation.  These pulsations are either driven waves that propogate across the surface of a 
star due to pressure and/or gravitational gradients, known as p-mode and g-mode waves respectively. These pulsation waves can be modeled by oscillating
spherical harmonics.  Using the "STARRY" python package one can model differential magnitudes of a spherical surface. However, there 
are no methods within this package enabling these surfaces to evolve or vary over time-- or in another word, pulsate. This program
utilizes these static maps to create a periodically pulsating stellar source by summing various spherical harmonic magnitude values
over time.  
"""

### IMPORT PACKAGES ###
import sys
import os
# sys.path.insert(0, '/Users/aayala/GitHub/starry')
import starry
starry.config.lazy = False
starry.config.quiet = True

import numpy as np
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import lightkurve as lk
from tqdm import tqdm
# import lmfit
from PIL import Image
import warnings
#import healpy

##Get/Set naming conventions

### Calculate position in CoeffArray of given L,M index

#Function to determine postition of desire spherical harmonic mode coefficient in L-M array
def lmIndex(l, m):
    return l*(l+1) + m

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
    posCoeff = amp*np.cos(2*np.pi*(frequency*(t+phase0)))
    negCoeff = -np.sign(m) * amp*np.sin(2*np.pi*(frequency*(t+phase0)))
    return posCoeff, negCoeff

#Star class object to function as source for stellar pulsation
class star:

    """Initialize STAR object to simulate stellar pulsation

    ### Parameters

    lmMode : array_like (int, 2D)
        Array of L-M modes of stellar pulsation
    freq : array_like (float)
        Pulsation frequencies of respective L-M modes in lmMode
    amp : array_like (float)
        Pulsation amplitudes of respective L-M modes in lmMode
    phase : array_like (float)
        Initial phase of L-M mode pulsations in lmMode
    inc : float (default = 90.0)
        Inclination of star object relative to observer
    lMax : int (default = None)
        Maximum L-mode complexity of star object
    fcn : float function (default = None)
        Set function for transforming star object surface map values
    osParam : int (default = 2)
        Integer to determine number of pixels produced from surface map transform
    observed : bool (default = True)
        Boolean flag of amplitudes being either observed values or intrinsic values
    
    ### Returns

    star : tuple
        Star object with pulsation modes, frequencies, and amplitudes as indicated by user

    ### Example
    
    """

    #Init function taking freq, amp, etc. to initialize star with features
    def __init__(self, lmMode, freq, amp, phase, inc=90, lMax = None, fcn = None, osParam = 2, observed = True):
        #Insert warning if fcn given and no lMax provided (make sure its large enough)
        #See if we can change complexity (lMax) of map after initialization
        self.lmMode = lmMode
        self.freq = freq
        self.amp = amp
        self.phase = phase
        self.inc = inc
        self.fcn = fcn
        self.observed = observed

        #Save Y2P from pixel transform as feature of star
        
        #Provided lMax means we will transform surface output values SH-maps.  Need to carry along larger transform array.
        if(lMax is None):
            if(len(lmMode) > 0):
                self.lMax = np.max(self.lmMode)
            else:
                self.lMax = 0
        else:
            self.lMax = lMax


        self.nSignals = len(self.lmMode)
        self._map = starry.Map(ydeg=self.lMax, amp=1.0, inc = self.inc - 90)
        #Look into creating inclination function

        self._computeAmpCoeffs() #Think about renaming
        #self.computeMapCoeffs()
        #self._computeFlux()

        if self.fcn is not None:
            self.setTransFcn(self.fcn, osParam)
            self.lat, self.lon, self.Y2P, self.P2Y, self.Dx, self.Dy = self._map.get_pixel_transforms(oversample=osParam)

    #Transform constructed stellar surface map to new values in accordance with input transform function
    def setTransFcn(self, fcn):
        """Set function for transforming surface map pixel magnitudes to desired values

        ### Parameters

        fcn : float (default = 1.0)
            Equation to transform surface magnitudes to observable values
        osParam: int (default = 2)
            Degree to which surface map of star will be granulated to pixels. Higher value equals more pixels

        ### Example
        
        """

        #Translate lons/lats due to inclination of star
        self.fcn = fcn
        
        # self.lat = self.lat-self.inc

    #Perform coefficient transform to retrieve necessary input amplitude map values in order to receive desired output amplitudes
    def _computeAmpCoeffs(self):
        """Compute the amplitude coefficients for converting map values to desired observables

        ### Parameters

        ### Example
        Add example
        """
        # DESIRED amplitudes set to ampScaleFactor, transform if observed flag is true
        self.ampScaleFactor = np.ones(self.nSignals)
        self.phaseOffsetArray = np.zeros(self.nSignals)

        #Sample map over arbitrary time to retrieve necessary amplitudes and phase to combine for constructing desired surfacd map
        for i in range(len(self.lmMode)):
            timeSample = np.arange(0,1,0.25)
            testFluxArray = np.zeros(len(timeSample))
            for j, t in enumerate(timeSample):
                self._map.y[1:] = 0.0
                l = self.lmMode[i][0]
                m = self.lmMode[i][1]
                posC,negC = _LxMx(t, m, frequency=1, amp=1.0, phase0=0)
                self._map[l,np.abs(m)]  += posC
                if m != 0:
                    self._map[l,-np.abs(m)] += negC

                testFluxArray[j] = self._map.flux()

            #Save amp and phase coefficients for use in map construction
            maxAmp = np.nanmax(testFluxArray)
            maxAmp = maxAmp-1.0
            if self.observed:
                self.ampScaleFactor[i] = 1.0/maxAmp
                if(self.ampScaleFactor[i] > 1.0):
                    warnings.warn("WARNING: Producing unphysical amplitude values!")
                    
            self.phaseOffsetArray[i] = (timeSample[np.argmax(testFluxArray)]-0.25)

    #Compute coefficients necessary to construct surface maps for pulsation over given time sample
    #Compute coefficients necessary to construct surface maps for pulsation over given time sample
    def computePulsation(self, time):
        """Construct surface maps for every timestamp in the given time period

        ### Parameters

        timeArray : array_like (float, 1D)
            Time values over which to construct surface maps

        ### Example
        
        """

        #Remove saving pos/neg Coeffs to self._map.  Store directly in self.coeffArray

        if not hasattr(time, '__iter__'):
            time = [time]
        
        coeffArray = np.zeros((len(time),len(self._map.y)))
        coeffArray[:,0] = 1.0
        for i,t in enumerate(time):
            for j in range(self.nSignals):
                    l = self.lmMode[j][0]
                    m = self.lmMode[j][1]
                    posC,negC = _LxMx(t, m, frequency=self.freq[j],amp=self.amp[j]*self.ampScaleFactor[j], 
                                        phase0=self.phase[j]+self.phaseOffsetArray[j])
                    
                    #print(posC)

                    coeffArray[i,lmIndex(l, np.abs(m))] += posC
                    if m != 0:
                        coeffArray[i,lmIndex(l,-np.abs(m))] += negC

            if self.fcn is not None:
                p = self.Y2P.dot(coeffArray[i,:])
                newP = self.fcn(p)
                coeffArray[i,:] = self.P2Y.dot(newP)

        return coeffArray

    #Calculate flux of surface map pulsation over given time sample (time Array given HERE)
    def computeFlux(self, time, binaryIndicator=False):
        """Compute output flux of star object over input time array and save as feature of star

        ### Parameters

        timeArray : array_like (float, 1D)
            Time values over which to construct surface maps

        binaryIndicator : boolean (deafault = False)
            Flag indicating whether star is in binary system
        
        
        ### Returns

        fluxArray : array_like (float, 1D)
            Integrated disc flux values of star at each value of timeArray 

        ### Example
        
        """
        coeffArray = self.computePulsation(time)
        fluxArray = np.zeros(len(time))


        if binaryIndicator == True:
            for i,t in enumerate(time):
                self.sys.primary.map.y[1:] = coeffArray[i][1:]
                fluxArray[i] = self.sys.flux(t)

        else:
            for i in range(len(time)):
                self._map.y[:] = coeffArray[i,:]
                fluxArray[i] = self._map.flux()
        
        return fluxArray
    
    #Internal starry dummy object
    
    #Construct binary system model using stellar pulsation source as primary and black source as secondary
    def setBinarySystem(self, r1, m1, r2, m2, sbRatio, period, t0):
        """Inserts star within binary system with given paramater inputs

        ### Parameters

        r1 : float
            Radius of primary star

        m1 : float
            Mass of primary star

        r2 : float
            Radius of secondary star

        m2 : float
            Mass of secondary star

        sbRatio : float (Must be positive or 0)
            Value of magnitude ratio of fluxes between primary and secondary star.  Higher value equals brighter secondary

        
        ### Returns

        ### Example
        
        """

        pri = starry.Primary(self._map, r=r1, m=m1, prot=np.inf)
            #starry.Primary(starry.Map(ydeg=primary.ydeg, inc=primary.inc, amp=amp1), r=r1, m=m1, prot=np.inf)
        sec = starry.Secondary(starry.Map(ydeg=0, inc=0, amp=sbRatio), r=r2, m=m2, porb=period, prot=np.inf, t0=t0, inc=90)

        self.sys = starry.System(pri, sec)



    #Function for computing flux map for SPECIFIC time instance and outputs map
    # def setTime(self, time, binaryFlag):
    #     """Construct surface map at specific time isntance and associate time value as a feature of star

    #     ### Parameters

    #     time : float
    #         Time value last associated with given star object

    #     binaryFlag : boolean (deafault = False)
    #         Flag indicating whether star is in binary system
        
        
    #     ### Returns

    #     ### Example
    #     Returns the desired output
    #     """

    #     _ = self.getFlux([time], binaryIndicator=binaryFlag)
    #     #self.time = time

    #Visualize should take a time parameter
    def visualizeStar(self):
        """Construct animated gif visualizing star's pulsation over specific time period

        ### Parameters
        
        ### Returns

        ### Example
        
        """
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.axis('off')

#         norm = np.linalg.norm(fluxArray)
#         normFluxArray = fluxArray/norm
        # print(np.min(normFluxArray))
        # print(np.max(normFluxArray))
        #vRange = np.nanmax(np.abs(self.coeffArray.flatten() - 1/np.pi))
        #vmid = 1.0/np.pi

        # Render the surface values first
        rendered = []
        for coeffs in tqdm(self.coeffArray[:,:]):
            self._map.y[:] = coeffs
            rendered.append(self._map.render())

        # Find most extreme surface brightness excursions for color bar
        vRange = np.nanmax(np.abs(np.array(rendered).flatten() - 1/np.pi))
        vmid = 1.0/np.pi

        # Plot individual frames for animation
        imList = []
        for render in rendered:
            im = ax.imshow(render, cmap="seismic_r", animated=True, vmin = vmid-vRange, vmax = vmid+vRange)
            imList.append([im])

        anim = animation.ArtistAnimation(fig, imList, interval = 50, blit=True)
        writergif = animation.PillowWriter(fps=30)
        anim.save('Pulsation.gif',writer=writergif)
        display(HTML(anim.to_html5_video()))

    def visualizeBinary(self):
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1,1,1)
        ax.axis('off')

        rendered = []
        for coeffs in tqdm(self.coeffArray[:,:]):
            self._map.y[:] = coeffs
            rendered.append(self._map.render())

        # Find most extreme surface brightness excursions for color bar
        vRange = np.nanmax(np.abs(np.array(rendered).flatten() - 1/np.pi))
        vmid = 1.0/np.pi
        fileNames = []
        
        for i in range(len(self.time)):
            fileLength = int(np.ceil(np.log10(len(self.time))))
            self.sys.primary.map.y[:] = self.coeffArray[i,:]
            fileNames.append(f"./BinaryImages/{i:0{fileLength}}.png")
            self.sys.show(t=self.time[i], figsize=(5, 5), cmap="seismic_r", file=fileNames[i])

        
        frames = [Image.open(fileName).convert("RGBA") for fileName in fileNames]
        newFrames = []

        for frame in frames:
            newFrame = Image.new("RGBA", frames[0].size, "WHITE")
            newFrame.paste(frame,(0,0), frame)
            newFrame.convert('RGB')
            newFrames.append(newFrame)

        newFrames[0].save("Binary.gif", format="GIF", append_images=newFrames, save_all=True, duration = 50, loop=0)

        # anim = animation.ArtistAnimation(fig, frames, interval = 50, blit=True)
        # writergif = animation.PillowWriter(fps=30)
        # anim.save('Pulsation.gif',writer=writergif)
        # display(HTML(anim.to_html5_video()))



### OLD STUFF (MAYBE DELETE) ###
"""

# Loop to create true flux array using newly determined phase offset values
for i,time in enumerate(timeArray):
    map.y[1:] = 0
    for j in range(len(freq)):
            l = lmMode[j][0]
            m = lmMode[j][1]
            posC,negC = _LxMx(time, m, frequency=freq[j],amp=amp[j]*ampScaleFactor[j], phase0=phase[j]+phaseOffsetArray[j])

            map[l,np.abs(m)]  += posC
            if m != 0:
                map[l,-np.abs(m)] += negC
            
    fluxArray[i] = map.flux()
    mapArray[i] = map.render()
    coeffArray[i] = map.y
"""
