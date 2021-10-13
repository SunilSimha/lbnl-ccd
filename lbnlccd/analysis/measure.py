import numpy as np

from astropy.nddata import CCDData
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


from ccdproc.combiner import Combiner
from ccdproc import ccdmask

from scipy.signal import correlate2d

# TODO: Replace the hard-coded region definitions with
# those from the fits headers.

QUAD_IDX = {'ll':0, 'lr':1, 'ul':2, 'ur':3}

# Horizontal x Vertical
def sampling_bounds(binning:str="1x1",size:int=300)->list:
    """
    A helper function to define the sampling region
    for analysis in all four quadrants. Returns
    fours tuples of pixel bounds (xmin, xmax, ymin, ymax),
    i.e. one for each CCD quadrant.
    """
    center_pixels = {'1x1':{'ll':[1100, 1100],
                            'lr':[3200, 1100],
                            'ul':[1100, 3200],
                            'ur':[3200, 3200]},
                     '2x1':{ 'll':[550, 1100],
                             'lr':[1600,1100],
                             'ul':[550, 3200],
                             'ur':[1600,3200]},
                     '1x2':{ 'll':[1100, 550],
                             'lr':[3200, 550],
                             'ul':[1100, 1600],
                             'ur':[3200, 1600]},
                     '2x2':{ 'll':[850, 850],
                             'lr':[1600, 850],
                             'ul':[850, 1300],
                             'ur':[1600, 1300]},
                    }
    middle = center_pixels[binning]
    ll_s = np.array([middle['ll'][0]-int(size/2), middle['ll'][0]+int(size/2),
            middle['ll'][1]-int(size/2), middle['ll'][1]+int(size/2)])
    lr_s = np.array([middle['lr'][0]-int(size/2), middle['lr'][0]+int(size/2),
            middle['lr'][1]-int(size/2), middle['lr'][1]+int(size/2)])
    ul_s = np.array([middle['ul'][0]-int(size/2), middle['ul'][0]+int(size/2),
            middle['ul'][1]-int(size/2), middle['ul'][1]+int(size/2)])
    ur_s = np.array([middle['ur'][0]-int(size/2), middle['ur'][0]+int(size/2),
            middle['ur'][1]-int(size/2), middle['ur'][1]+int(size/2)])
    
    return np.array([ll_s, lr_s, ul_s, ur_s])

def random_sampling_bounds(binning:str="1x1", size=300, max_offset:int=30):
    """
    Randomizes the centers of the sampling bounds
    returned by the sampling_bounds function.
    """
    x_off, y_off = np.random.randint(-int(max_offset), int(max_offset),2)
    bounds = sampling_bounds(binning, size)
    for b in bounds:
        b += np.array([x_off, x_off, y_off, y_off])
    return bounds
    
# Bounds for the vertical overscan region
# bounds are given as [xmin, xmax, ymin, ymax] for 
# all binning modes.
def get_os_bounds(binning:str="1x1"):
    os_regions = {'1x1':{'ll':[2065, 2117, 0, 2084],
                        'lr':[2118, 2170, 0, 2084],
                        'ul':[2065, 2117, 2085, 4247],
                        'ur':[2118, 2170, 2085, 4247]},
                '2x1':{'ll':[1032, 1103, 0, 2084],
                        'lr':[1104, 1176, 0, 2084],
                        'ul':[1032, 1103, 2085, 4247],
                        'ur':[1104, 1176, 2085, 4247]},
                '1x2':{'ll':[2065, 2117, 0, 1092],
                        'lr':[2118, 2170, 0, 1092],
                        'ul':[2065, 2117, 1093, 2184],
                        'ur':[2350, 2350, 1093, 2184]},
                '2x2':{'ll':[1032, 1103, 0, 1092],
                        'lr':[1104, 1176, 0, 1092],
                        'ul':[1032, 1103, 1093, 2184],
                        'ur':[1104, 1176, 1093, 2184]},
                    }
    os_bounds = os_regions[binning]
    ll_b = os_bounds['ll']
    lr_b = os_bounds['lr']
    ul_b = os_bounds['ul']
    ur_b = os_bounds['ur']
    return ll_b, lr_b, ul_b, ur_b

def make_badpix_map(flat_s:str, flat_l:str, loc:str="badpix.fits"): 
    ccd1 = CCDData.read(flat_s) 
    ccd2 = CCDData.read(flat_l) 
    ratio = ccd2.divide(ccd1) 
    maskr = ccdmask(ratio) 
    hdu = fits.PrimaryHDU(maskr.astype(int)) 
    hdulist = fits.HDUList([hdu]) 
    hdulist.writeto(loc, overwrite=True) 
    return

def median_combine(frames:list, out:str, overwrite:bool=False): 
    ccd_data_list = [] 
    for frame in frames:
        hdulist = fits.open(frame)
        ccddata_inst = CCDData(hdulist[0].data, meta=hdulist[0].header,
                                unit="adu", mask=None, uncertainty=None)
        ccd_data_list.append(ccddata_inst) 
    combiner = Combiner(ccd_data_list) 
    med = combiner.median_combine()
    med.write(out, overwrite=overwrite)
    return med

def get_readnoise(framedata, binning='1x1'):
    """
    Estimates the standard deviation of pixels
    in the overscan regions for all four quadrants.
    """
    ll_b, lr_b, ul_b, ur_b = get_os_bounds(binning)

    _,_, ll_rn = sigma_clipped_stats(framedata[ll_b[2]:ll_b[3], ll_b[0]:ll_b[1]])
    _,_, ul_rn = sigma_clipped_stats(framedata[lr_b[2]:lr_b[3], lr_b[0]:lr_b[1]])
    _,_, lr_rn = sigma_clipped_stats(framedata[ul_b[2]:ul_b[3], ul_b[0]:ul_b[1]])
    _,_, ur_rn = sigma_clipped_stats(framedata[ur_b[2]:ur_b[3], ur_b[0]:ur_b[1]])
    return np.array([ll_rn, lr_rn, ul_rn, ur_rn])

def subtract_overscan(frame:str, median:bool=True,
                      write:bool=False,out:str=None, binning:str='1x1'):

    # Define overscan bounds
    ll_b, lr_b, _, _ = get_os_bounds(binning)

    # Read data
    if type(frame)==str:
        ccdframe = fits.getdata(frame,hdu=0)
    else:
        ccdframe = frame
    # Identify the overscan region
    #os_left = ccdframe.data[:, 2068:2200]
    #os_right = ccdframe.data[:, 2201:2333]
    os_left = ccdframe[:, ll_b[0]:ll_b[1]]
    os_right = ccdframe[:, lr_b[0]:lr_b[1]]  
    
    # Get a central bias value corresponding to each row
    scs_left = sigma_clipped_stats(os_left, axis=1)
    scs_right = sigma_clipped_stats(os_right, axis=1)
    
    if median:
        b_left = scs_left[1]
        b_right = scs_right[1]
    else: # Take mean
        b_left = scs_left[0]
        b_right = scs_right[0]

    # subtract
    newframe = CCDData(np.zeros(ccdframe.shape), unit="adu")
    #newframe.data[:,:2200] = (ccdframe.data[:,:2200].T- b_left).T
    #newframe.data[:,2201:] = (ccdframe.data[:,2201:].T- b_right).T
    newframe.data[:, :ll_b[1]] = (ccdframe[:, :ll_b[1]].T- b_left).T
    newframe.data[:, lr_b[0]:] = (ccdframe[:, lr_b[0]:].T- b_right).T
    
    if write:
        if out is None:
            out = frame[:-5]+"_ossub.fits"
        newframe.write(out, overwrite=True)

    return newframe

def quad_measure(framedata, func, binning="1x1", sample_size=300, randomize=False, max_offset=30):
    """
    Apply a function on all quadrants
    and return value in the following order
    LL, LR, UL, UR
    """
    if randomize:
        ll_s, lr_s, ul_s, ur_s = random_sampling_bounds(binning, sample_size, max_offset)
    else:
        ll_s, lr_s, ul_s, ur_s = sampling_bounds(binning, sample_size)
    #ll = func(framedata[1290:1554,1294:1558])
    #lr = func(framedata[1290:1554,2844:3108])
    #ul = func(framedata[2848:3112,1294:1558])
    #ur = func(framedata[2848:3112,2844:3108])
    ll = func(framedata[ll_s[2]:ll_s[3],ll_s[0]:ll_s[1]])
    lr = func(framedata[lr_s[2]:lr_s[3],lr_s[0]:lr_s[1]])
    ul = func(framedata[ul_s[2]:ul_s[3],ul_s[0]:ul_s[1]])
    ur = func(framedata[ur_s[2]:ur_s[3],ur_s[0]:ur_s[1]])

    return np.array([ll, lr, ul, ur])

def generate_diff_img(flatframe1:str, flatframe2:str)->np.ndarray:
    """
    Return the difference of two image arrays.
    """
    fl_1 = fits.getdata(flatframe1, hdu=0)
    fl_2 = fits.getdata(flatframe2, hdu=0)
    diff = fl_1.astype(np.float64) - fl_2.astype(np.float64)
    return diff

def flat_pix_corr(diff:np.ndarray, quadrant:str="ll",
                  binning:str="1x1", sample_size=300, randomize:bool=False,
                  max_offset:int=30)->np.ndarray:
    """
    Compute the correlation matrix for a given flat image difference in
    a quadrant of your choice.
    """
    if randomize:
        l1, l2, l3, l4 = random_sampling_bounds(binning, sample_size, max_offset)[QUAD_IDX[quadrant]]
    else:
        l1, l2, l3, l4 = sampling_bounds(binning, sample_size)[QUAD_IDX[quadrant]]
    chunk = diff[l3:l4,l1:l2]
    R_matrix = correlate2d(chunk, chunk, mode="same")/np.sum(chunk**2)
    R_matrix[int((l4-l3)/2)-1, int((l2-l1)/2)-1] -= 1.0

    return R_matrix

def gain_measurement(flat1:str, flat2:str, dark:str, return_gain:bool = False,
                     sample_size:int=300,binning:str="1x1"):
    """
    Given a flat frame and its corresponding
    dark frame, measure the gain  for each amplifier.
    """

    # Read in frames. These should be bias corrected.
    flatframe_1 = fits.getdata(flat1,dtype=np.int16)
    flatframe_2 = fits.getdata(flat2,dtype=np.int16)
    darkframe = fits.getdata(dark,dtype=np.int16)

    # Create a bias subtracted dark
    ossub_dark = subtract_overscan(dark,binning=binning)
    #import pdb; pdb.set_trace()
    # Measure signal and noise
    noise_img = flatframe_1.astype(np.int64)-flatframe_2.astype(np.int64)
    s = quad_measure(flatframe_1.astype(np.int64)-darkframe.astype(np.int64), sigma_clipped_stats, binning=binning, sample_size=sample_size)[:,0]
    sigma = quad_measure(noise_img, sigma_clipped_stats, binning=binning, sample_size=sample_size)[:,2]
    rn_array = get_readnoise(noise_img, binning=binning)
    #var_dark = quad_measure(ossub_dark.data, sigma_clipped_stats, binning=binning)[:,0] # Assuming poisson stats, the variance is the same as the mean.

    var_s = (sigma**2-rn_array**2)/2#-var_dark)/2

    # Do you want the gain or the median and variance separately?
    if return_gain:
        return s/var_s
    else:
        return s, var_s

def ptc(flat1_list:list, flat2_list:list, dark_list:list, sample_size:int=300, binning:str="1x1")->list:

    s_list = np.zeros((len(flat1_list),4))
    var_s_list = np.zeros_like(s_list)
    #import pdb; pdb.set_trace()
    for num, (flat1, flat2, dark) in enumerate(zip(flat1_list, flat2_list, dark_list)):
        s, var_s = gain_measurement(flat1, flat2, dark, False, binning=binning, sample_size=sample_size)
        s_list[num] = s
        var_s_list[num] = var_s
    
    return s_list, var_s_list

def cte_eper(eper_flat:str):

    flatframe = CCDData.read(eper_flat, unit="adu")

    # Parallel CTE

    # Define region of interest
    # LL
    y1 = 1132
    x1 = 2067
    y2 = 3268
    x2 = 2333

    # Regions with the parallel overscan
    LL = flatframe.data[y1-600:y1, x1-500:x1]
    LR = flatframe.data[y1-600:y1, x2:x2+500]
    UL = flatframe.data[y2:y2+600, x1-500:x1]
    UR = flatframe.data[y2:y2+600, x2:x2+500]
    regions = [LL, LR, UL, UR]

    # Get sigma clipped stats:
    means = []
    stds = []
    for region in regions:
        scs = sigma_clipped_stats(region, axis=1)
        means.append(scs[0])
        stds.append(scs[2])
    
    # Estimate bias and subtract
    cte_p = []
    cte_p_err = []
    for num,(mean_arr,std_arr) in enumerate(zip(means,stds)):
        # For lower regions
        if num<2:
            bias, _, bias_err = sigma_clipped_stats(mean_arr[-105:])
            bias_sub = mean_arr-bias
            Sd = bias_sub[-100]
            dSd = np.sqrt(std_arr[-100]**2+bias_err**2)
            Slc = bias_sub[-101]
            dSlc = np.sqrt(std_arr[-101]**2+bias_err**2)
            Np = y1-100
        else:
            bias = sigma_clipped_stats(mean_arr[:95])[0]
            bias_sub = mean_arr-bias
            Sd = bias_sub[99]
            dSd = np.sqrt(std_arr[99]**2+bias_err**2)
            Slc = bias_sub[100]
            dSlc = np.sqrt(std_arr[100]**2+bias_err**2)
            Np = 4400-100-y2
        
        cte_p.append(1-Sd/Slc/Np)
        cte_p_err.append((dSd/Slc + Sd*dSlc/Slc**2)/Np)
    
    # Serial CTE

    # Go though the same steps
    # Note these regions are ordered differently than before. Now it is LL, UL, LR, UR
    LL = flatframe.data[y1-600:y1-100, x1-500:x1+100]
    UL = flatframe.data[y2+100:y2+600, x1-500:x1+100]
    LR = flatframe.data[y1-600:y1-100, x2-100:x2+500]
    UR = flatframe.data[y2+100:y2+600, x2-100:x2+500]
    regions = [LL, UL, LR, UR]

    # Get sigma clipped stats:
    means = []
    stds  = []
    for region in regions:
        scs = sigma_clipped_stats(region, axis=0)
        means.append(scs[0])
        stds.append(scs[2])
    
    # Estimate bias and subtract
    cte_s = []
    cte_s_err = []
    for num,(mean_arr,std_arr) in enumerate(zip(means,stds)):
        # For regions on the left
        if num<2:
            bias = sigma_clipped_stats(mean_arr[-95:])[0]
            bias_sub = mean_arr-bias
            Sd = bias_sub[-100]
            dSd = np.sqrt(std_arr[-100]**2+bias_err**2)
            Slc = bias_sub[-101]
            dSlc = np.sqrt(std_arr[-101]**2+bias_err**2)
            Np = y1-100
        # For regions on the right
        else:
            bias = sigma_clipped_stats(mean_arr[:95])[0]
            bias_sub = mean_arr-bias
            Sd = bias_sub[99]
            dSd = np.sqrt(std_arr[99]**2+bias_err**2)
            Slc = bias_sub[100]
            dSlc = np.sqrt(std_arr[100]**2+bias_err**2)
            Np = 4400-x2
        cte_s.append(1-Sd/Slc/Np)
        cte_s_err.append((dSd/Slc + Sd*dSlc/Slc**2)/Np)
    
    return np.array(cte_p), np.array(cte_p_err), np.array(cte_s)[[0,2,1,3]],np.array(cte_s_err)[[0,2,1,3]] # reorder CTE_s