#Decription:
"""
This program creates a simulation of periodic non-radial stellar pulsation.  These pulsations are waves that propogate across the surface of a 
star due to either pressure or gravitational gradients, known as p-mode and g-mode waves respectively. These pulsation waves can be modeled by 
spherical harmonics.  Using the "Starry" python package one can model differential magnitudes of a spherical surface. However, there 
are no methods within this package enabling these surfaces to evolve or vary over time-- or in another word, pulsate. My program attempts 
to exploit the static nature of Starry to create a periodically pulsating stellar source by summing various spherical harmonic magnitude values
over time.  
"""

### IMPORT PACKAGES ###
import numpy as np
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
import starry
import matplotlib.animation as animation
from IPython.display import HTML
starry.config.lazy = False
starry.config.quiet = True
import lightkurve as lk
from tqdm import tqdm
import lmfit
#import healpy

#Function to calculate coefficients to construct standing wave pulsation from combining +/- m-modes
def LxMx(t, m, frequency=1,amp=1, phase0=0):
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
    posCoeff = amp*np.cos(2*np.pi*(frequency*t+phase0))
    negCoeff = -np.sign(m) * amp*np.sin(2*np.pi*(frequency*t+phase0))
    return posCoeff, negCoeff

#Star class object to function as source for stellar pulsation
class star:

    """Initialize STAR object to simulate stellar pulsation

    ### Parameters

    lmArray : array_like (int, 2D)
        Array of L-M modes of stellar pulsation
    fArray : array_like (float)
        Pulsation frequencies of respective L-M modes in lmArray
    aArray : array_like (float)
        Pulsation amplitudes of respective L-M modes in lmArray
    phaseArray : array_like (float)
        Initial phase of L-M mode pulsations in lmArray
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
    def __init__(self, lmArray, fArray, aArray, phaseArray, inc=90, lMax = None, fcn = None, osParam = 2, observed = True):
        #Insert warning if fcn given and no lMax provided (make sure its large enough)
        #See if we can change complexity (lMax) of map after initialization
        self.lmArray = lmArray
        self.fArray = fArray
        self.aArray = aArray
        self.phaseArray = phaseArray
        self.inc = inc
        self.fcn = fcn
        self.observed = observed
        self.time = 0.0
        self.flux = np.zeros(1)
        self.timeArray = np.zeros(1)
        

        if self.fcn is not None:
            self.setTransFcn(self.fcn, osParam)
        
        #Provided lMax means we will transform surface output values SH-maps.  Need to carry along larger transform array.
        if(lMax is None):
            if(len(lmArray) > 0):
                self.lMax = np.max(self.lmArray)
            else:
                self.lMax = 0
        else:
            self.lMax = lMax


        self.nSignals = len(self.lmArray)
        self.map = starry.Map(ydeg=self.lMax, amp=1.0)
        self.map.inc = 90 - self.inc
        #Look into creating inclination function

        self._computeAmpCoeffs()

    #Transform constructed stellar surface map to new values in accordance with input transform function
    def setTransFcn(self, fcn, osParam = 2):
        """Set function for transforming surface map pixel magnitudes to desired values

        ### Parameters

        fcn : float (default = 1.0)
            Equation to transform surface magnitudes to observable values
        osParam: int (default = 2)
            Degree to which surface map of star will be granulated to pixels. Higher value equals more pixels

        ### Example
        
        """
        self.lat, self.lon, self.Y2P, self.P2Y, self.Dx, self.Dy = self.map.get_pixel_transforms(oversample=osParam)
        self.fcn = fcn

    #Perform coefficient transform to retrieve necessary input amplitude map values in order to receive desired output amplitudes
    def _computeAmpCoeffs(self):
        """Compute the amplitude coefficients for converting map values to desired observables

        ### Parameters

        ### Example
        
        """
        # DESIRED amplitudes set to ampCoeffArray, transform if observed flag is true
        self.ampCoeffArray = np.ones(self.nSignals)
        self.phaseOffsetArray = np.zeros(self.nSignals)

        #Sample map over arbitrary time to retrieve necessarry amplitudes and phases to combine for constructing desired surfacd map
        for i in range(len(self.lmArray)):
            timeSample = np.arange(0,1,0.25)
            testFluxArray = np.zeros(len(timeSample))
            for j, t in enumerate(timeSample):
                self.map.y[1:] = 0
                l = self.lmArray[i][0]
                m = self.lmArray[i][1]
                posC,negC = LxMx(t, m, frequency=1, amp=1.0, phase0=0)
                self.map[l,np.abs(m)]  += posC
                if m != 0:
                    self.map[l,-np.abs(m)] += negC

                testFluxArray[j] = self.map.flux()

            #Save amp and phase coefficients for use in map construction
            maxAmp = np.nanmax(testFluxArray)
            maxAmp = maxAmp-1.0
            if self.observed:
                self.ampCoeffArray[i] = 1.0/maxAmp
            self.phaseOffsetArray[i] = (timeSample[np.argmax(testFluxArray)]-0.25)

    #Compute coefficients necessary to construct surface maps for pulsation over given time sample
    def _computeMapCoeffs(self, timeArray):
        """Construct surface maps for every timestamp in the given time period

        ### Parameters

        timeArray : array_like (float, 1D)
            Time values over which to construct surface maps

        ### Example
        
        """
        self.timeArray = timeArray
        self.coeffArray = np.zeros((len(timeArray), len(self.map.y)))
        for i,time in enumerate(timeArray):
            self.map.y[0] = 1.0
            self.map.y[1:] = 0
            for j in range(self.nSignals):
                    l = self.lmArray[j][0]
                    m = self.lmArray[j][1]
                    posC,negC = LxMx(time, m, frequency=self.fArray[j],amp=self.aArray[j]*self.ampCoeffArray[j], 
                                     phase0=self.phaseArray[j]+self.phaseOffsetArray[j])

                    self.map[l,np.abs(m)]  += posC
                    if m != 0:
                        self.map[l,-np.abs(m)] += negC
            self.coeffArray[i,:] = self.map.y
            ###Insert function transform
            if self.fcn is not None:
                p = self.Y2P.dot(self.map.y)
                newP = self.fcn(p)
                self.coeffArray[i,:] = self.P2Y.dot(newP)
                #print(p, newP, self.coeffArray[i])


    #Calculate flux of surface map pulsation over given time sample (time Array given HERE)
    def computeFlux(self, timeArray, binaryIndicator=False):
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
        self._computeMapCoeffs(timeArray)
        fluxArray = np.zeros(len(timeArray))


        if binaryIndicator == True:
            for i,t in enumerate(timeArray):
                self.sys.primary.map.y[1:] = self.coeffArray[i][1:]
                fluxArray[i] = self.sys.flux(t)

        else:
            for i,time in enumerate(timeArray):
                self.map.y[:] = self.coeffArray[i,:]
                self.time = time
                fluxArray[i] = self.map.flux()
        
        self.flux = fluxArray
        #self.time = timeArray[-1]

        return fluxArray
    
    #Construct binary system model using stellar pulsation source as primary and black source as secondary
    def binarySystem(self, r1, m1, r2, m2, sbRatio, period, t0):
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

        pri = starry.Primary(self.map, r=r1, m=m1, prot=np.inf)
            #starry.Primary(starry.Map(ydeg=primary.ydeg, inc=primary.inc, amp=amp1), r=r1, m=m1, prot=np.inf)
        sec = starry.Secondary(starry.Map(ydeg=0, inc=0, amp=sbRatio), r=r2, m=m2, porb=period, prot=np.inf, t0=t0, inc=90)

        self.sys = starry.System(pri, sec)


    #Function for computing flux map for SPECIFIC time instance and outputs map
    def setTime(self, time, binaryFlag):
        """Construct surface map at specific time isntance and associate time value as a feature of star

        ### Parameters

        time : float
            Time value last associated with given star object

        binaryFlag : boolean (deafault = False)
            Flag indicating whether star is in binary system
        
        
        ### Returns

        ### Example
        
        """

        _ = self.computeFlux([time], binaryIndicator=binaryFlag)
        #self.time = time

    def visualize(self):
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
        for coeffs in tqdm(self.coeffArray[::5]):
            self.map.y[:] = coeffs
            rendered.append(self.map.render())

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


### OLD STUFF (MAYBE DELETE) ###
"""

# Loop to create true flux array using newly determined phase offset values
for i,time in enumerate(timeArray):
    map.y[1:] = 0
    for j in range(len(fArray)):
            l = lmArray[j][0]
            m = lmArray[j][1]
            posC,negC = LxMx(time, m, frequency=fArray[j],amp=aArray[j]*ampCoeffArray[j], phase0=phaseArray[j]+phaseOffsetArray[j])

            map[l,np.abs(m)]  += posC
            if m != 0:
                map[l,-np.abs(m)] += negC
            
    fluxArray[i] = map.flux()
    mapArray[i] = map.render()
    coeffArray[i] = map.y
"""
