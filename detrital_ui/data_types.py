import numpy as np
import os, sys
import pickle
import common as ba
import pylab as plt

import ogr, gdal
from gdalconst import *
import numpy.ma as ma
from matplotlib.transforms import Affine2D

NBINS=512
SCALE = 0.001  # from UTM meters to km

class Catchment(object):
    def __init__(self, dem, name):
        gdal.AllRegister()
        self.img = gdal.Open(dem, GA_ReadOnly)
        self.name = name
        if self.img is None: sys.exit('Could not open ' + dem)
        if self.img.RasterCount > 1: sys.exit("Too many bands in the dem")
        if self.img.GetRasterBand(1).GetNoDataValue() is None: sys.exit("DEM does not have NODATA value specified")
        self._data = SCALE * self.img.GetRasterBand(1).ReadAsArray() 
        self._nodata = SCALE * self.img.GetRasterBand(1).GetNoDataValue()
        self._bins=None


    def _get_transform(self):
        '''Transfrom from pixel coordinates into geographic'''
        T = self.img.GetGeoTransform()
        T = Affine2D([[T[1],T[2],T[0]],[T[4],T[5],T[3]],[0,0,1]])
        T.scale(SCALE) # to km UNNECESSARY?
        return T


    @property
    def xyz(self):
        xs_i = range(self.img.RasterXSize)
        ys_i = range(self.img.RasterYSize)
        T = self._get_transform()
        x,y = np.meshgrid(T.transform(np.column_stack((xs_i,xs_i)))[:,0],
                          T.transform(np.column_stack((ys_i,ys_i)))[:,1])
        mask = (self._data != self._nodata)
        
        return np.column_stack((x[mask].flatten(),y[mask].flatten(),self._data[mask].flatten()))

    @property
    def center(self):
        return np.apply_along_axis(np.mean,0,self.xyz)
        
    def plot(self):
        data = ma.masked_values(self._data, self._nodata,copy=False)
        T = self._get_transform()
        plt.imshow(data,transform=T,  #transform argument doesn't seem to work
            extent=T.transform([[0, self.img.RasterYSize-1],
                                [self.img.RasterXSize-1, 0]]).T.flatten())
        x,y = self.center[0:2]
        plt.text(x,y,self.name)

    @property
    def bins(self):
        """Returns weights (probabilities) from a histogram of hypsometry"""
        #Use it to speed things up.
        if self._bins is None:
            print "WARNING: Binning hypsometry"
            h_w, h_edges = np.histogram(self.xyz[:,2], bins=NBINS, normed=False)
            h_w = h_w / float(self.xyz.shape[0])
            h_b  = h_edges[:-1] + np.diff(h_edges)/2  # Replace bin margins with bin centers
            self._bins = {'h':h_b, 'w':h_w}

        return self._bins


class Sample(object):
    """Base class for bedrock and detrital samples"""
    use=True  #temp. fix, move to Bedrock, once have time to re-cache
    def get_zrange(self):
        return np.linspace(-3,3)
        #return np.linspace(self.ref_z-4, 1.1 * max(self.z))
        
    def plot(self):
        pass

class DetritalSample(Sample):
    """
    use=False, sample will not be used to fit the model, but will be plotted (e.g. gof)
    """
    def __init__(self, age_file, sample_name, catchment, tc_type, use=True, **kwargs):
        self.use=use
        try:
            self.ages = np.genfromtxt(age_file, **kwargs)
        except IOError:
            print "\nThe specified file path or file name for ages is invalid: %s" % age_file
            sys.exit()
        if (self.ages.ndim > 1):
            print "\nDetrital age file %s has more than one column, specify usecols argument" % age_file
            sys.exit()
        
        if any(self.ages > 4500): sys.exit("Ages are not in My or older than the Earth.")
        
        self.name = sample_name
        self.catchment = catchment
        self.tc_type = tc_type

        #self.xyz = self.catchment.xyz
        #self.y = self.catchment.y
        #self.z = self.catchment.z
        
    @property
    def ref_z(self): return self.catchment.center[2]

                
class BedrockSample(Sample):
    @ba.autopickle
    def __init__(self, br_file, sample_name, tc_type, ref_z=None, catchment=None, **kwargs):
        '''
If catchment is provided, only samples that fall within the convex hull around the catchment will be taken.
        '''
        #self.use=use
        try:
            br_data = np.genfromtxt(br_file, names=True, dtype=None, **kwargs)
        except IOError:
            print "The specified file path or file name for bedrock data is invalid: %s" % br_file
            sys.exit()
        #The following assumes the bedrock data has columns named x, y, alt, age, and age_sd
        try:
            self.x = SCALE*br_data['x'] # Warning: assuming input is in meters
            self.y = SCALE*br_data['y']
        except ValueError:
            sys.exit("No coordinate data found")
            self.x,self.y = None,None
        try:
            self.z = br_data['alt'].astype(float)
            self.ages = br_data['age']
            self.ages_sd = br_data['age_sd']
        except ValueError:  
            sys.exit("The header of br_file is not in the proper format")
        
        #Convert elevations to kilometers
        if max(self.z) > 10:
            print "WARNING: Converting bedrock sample elevations to km"
            self.z /= 1000.0

        self.name = sample_name
        self.tc_type = tc_type
        
        self.ref_z = ref_z
        if self.ref_z is None:
            if catchment is None: self.ref_z = 0.5 * ( np.max(self.z) - np.min(self.z) )
            else:                 self.ref_z = catchment.center[2]

        if catchment is not None:
            print "Removing bedrock samples outside the catchment"
            poly = ba.convex_hull(catchment.xyz[:,0:2].T)
            idx = ba.points_in_poly(zip(self.x,self.y),poly)
            self.x=self.x[idx]  # Maybe change the way data is stored
            self.y=self.y[idx]
            self.z=self.z[idx]
            self.ages=self.ages[idx]
            self.ages_sd=self.ages_sd[idx]
            
    
    def getPickleFilename(self, br_file, sample_name, tc_type, ref_z=None, catchment=None, **kwargs):
        suffix=''
        if catchment is not None: suffix = '.'+catchment.name
        return "%s%s.cache" % (br_file,suffix)

    @property
    def xyz(self):
        return np.column_stack((self.x,self.y,self.z))
        
    def plot(self):
        ax = plt.gca()
        ax.scatter(self.x, self.y,color='k',marker='+')
        offset=0.5
        #import ipdb;ipdb.set_trace()
        for i,a in enumerate(self.ages):
            ax.text(self.x[i]+offset, self.y[i]+offset, '%.1f' % a, size='x-small')
        
        
        