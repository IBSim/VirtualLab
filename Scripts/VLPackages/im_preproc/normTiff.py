#!/usr/bin/env python3

import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal
import tifffile as tf
"""
Set of functions for image normalisation of tiff stack
"""
def normTiff(fname,**kwargs):
    '''
    Function to normalise CT-image data from a tiff file between pixel
    values for the materal and air.

    The function uses histogram values to determine 
    the average pixel values for air and the material.

    It then normalises the data based on those values.

    required parameters:

    param:  fname - path to tiff image file including the file extension

    Optional kwargs:

        nbins - number of bins used for the histograms. Default is 256

        des_bg - float to determine the pixel value for peak intensity 
                corresponding to air in the final image based on the 
                max of the given data (or 1.0 for floating point types).

                For int datatypes that is uint32 uint16 and uint8.
                des_bg=0.1 would mean the peak pixel value for air 
                would be set as 0.1*maxdtype rounded to the nearest 
                int. So for example with raw_type='uint16' this would be 
                0.1*65535 = 6553

                However, for float based datatypes i.e. float32 and float16. 
                Pixel values are assumed to be between 0.0 and 1.0. 

                Thus des_bg=0.1 would mean the peak pixel value for air 
                would be set as 0.1.

        des_fg - float to determine the pixel value for peak intensity 
                corresponding to material in the final image based on 
                the max of the given data (or 1.0 for floating point types). 

                For int datatypes, that is uint32, uint16 and uint8.
                des_fg = 0.9 would mean the peak pixel value for 
                material would be set as 0.9*maxdtype rounded to 
                the nearest int. So for example with raw_type='uint16'
                this would be 0.9*65535 = 58981

                However, for float based datatypes i.e. float32 and float16. 
                Pixel values are assumed to be between 0.0 and 1.0. 

                Thus des_fg=0.9 would mean the peak pixel value for air 
                would be set as 0.9.

        peak_width -    pixel width of the peaks used, default is 1

        set_order_no -  determines number of peaks that are used for 
                        comparison when determining air vs material.
                        See https://docs.scipy.org/doc/scipy/reference
                        /generated/scipy.signal.argrelextrema.html
                        for more info.
    '''

    default_kwargs ={
        'nbins':512,
        'des_bg':0.1,
        'des_fg':0.9,
        'peak_width':1, 
        'set_order_no':10,
    }
      
    kwargs = { **default_kwargs, **kwargs }

    nbins=kwargs['nbins']
    des_bg = kwargs['des_bg']
    des_fg = kwargs['des_fg']
    peak_width = kwargs['peak_width']  
    set_order_no=  kwargs['set_order_no']
    if not os.path.exists(fname):
        raise FileNotFoundError (f'The image file {fname} filename could not be found.')
    root, ext = os.path.splitext(fname)
    if not ext:
        raise ValueError(f'Invalid file name {fname} filename must include a file extension.')
    elif ext not in ['.tiff','.tif']:
        raise ValueError(f'Invalid input file {fname} file must be a tiff stack.')

    # Read data into array   
    img_arr = tf.imread(fname)
    dt = str(img_arr.dtype)
    dtype_min, dtype_max, pix_air, pix_material = check_valid_np_type(dt,des_bg,des_fg)
    # Create histogram to find peaks
    (n, bins) = np.histogram(img_arr, nbins, density=True)
    plt.plot(.5*(bins[1:]+bins[:-1]), n)
    plt.xlabel('x positions (bins)')
    plt.ylabel('y positions  (counts)')
    histo_x = bins
    histo_y=np.zeros(shape=(nbins+1))
    histo_y[1:nbins+1] = n

    # number of picks dependes on the order, 1,3=80, 2=75, 4=66
    indexes = scipy.signal.argrelextrema(
        np.array(histo_y), comparator=np.greater, order=set_order_no)
    print('Peaks are: %s' % (indexes[0]))

    # finding the pick point
    i = 0
    pick_point = [0 for x in range(len(indexes))]
    pick_point_x = [0 for x in range(len(indexes))]
    for x in indexes:
        pick_point[i] = histo_y[x]
        pick_point_x[i] = histo_x[x]
        i = i+1

    peaks = np.zeros(shape=(3,len(indexes[0])))
    peaks = peaks.astype(np.float32)
    peaks[0,:] = indexes[0]
    peaks[1,:] = pick_point[0]
    peaks[2,:] = pick_point_x[0]
    peakssort = peaks[:,peaks[1].argsort()]
    # accessing the peak location in x-axis
    pickpoint_x = pick_point_x[0]
    # mean calculation
    idx_y = indexes[0]
    air_idx = int(peakssort[0,len(indexes[0])-1])
    material_idx = int(peakssort[0,len(indexes[0])-2])
    desired_air = histo_x[air_idx]
    desired_material = histo_x[material_idx]
    # mean for air
    bin_min_air = abs(peak_width-air_idx)
    bin_max_air = (air_idx+peak_width)
    data_in_air_bin_range = histo_y[bin_min_air:bin_max_air]
    air_bin_range = histo_x[bin_min_air:bin_max_air]

    mean_air = sum(np.multiply(data_in_air_bin_range, air_bin_range)
                )/sum(data_in_air_bin_range)

    # mean for materials
    bin_min_material = abs(peak_width-material_idx)
    bin_max_material = (material_idx+peak_width)

    data_in_material_bin_range = histo_y[bin_min_material:bin_max_material]
    material_bin_range = histo_x[bin_min_material:bin_max_material]

    mean_material = sum(np.multiply(data_in_material_bin_range,
                                    material_bin_range))/sum(data_in_material_bin_range)

    print('mean_air')
    print(mean_air)
    print('mean_material')
    print(mean_material)

    # converting list to array
    pick_point = np.array(pick_point)
    pick_point_x = np.array(pick_point_x)

    plt.clf()
    plt.plot(histo_x, histo_y)                      # histogram of Normalized Image
    plt.plot(pick_point_x, pick_point,  'go')      # Draw all the peaks found
    # Highlight selected two peaks
    x_number_list = [peakssort[2,len(indexes[0])-1], peakssort[2,len(indexes[0])-2]]
    y_number_list = [peakssort[1,len(indexes[0])-1], peakssort[1,len(indexes[0])-2]]

    plt.plot(x_number_list, y_number_list, 'mo')
    # Display the labels
    plt.xlabel('x positions (bins)')
    plt.ylabel('y positions  (counts)')
    plt.title('Pixel Histogram of Input Image')
    plt.savefig(root+'_Input_hist.png')
    
    # Normalise data based on peak values
    # Convert into float32 to avoid data clipping
    img_arr = img_arr.astype(np.float32)
    # Normalise
    img_arr = pix_air+(pix_material-pix_air)*(img_arr-mean_air)/(mean_material-mean_air)
    # Set any values outside new range to min/max (i.e. over/under saturate)
    img_arr[img_arr < dtype_min] = dtype_min
    img_arr[img_arr > dtype_max] = dtype_max

    # Convert back into original format
    img_arr = img_arr.astype(dt)
    # Write normalised data to file
    im_output=f"{root}_N{ext}"
    tf.imwrite(im_output,img_arr,bigtiff=True)

    # Generate new histogram with normalised data for comparison
    (n_norm, bins_norm) = np.histogram(img_arr, nbins, density=True)
    histo_x_norm = bins_norm
    histo_y_norm=np.zeros(shape=(nbins+1))
    histo_y_norm[1:nbins+1] = n_norm

    #plt.figure()
    plt.plot(histo_x_norm, histo_y_norm, color='red', linestyle='dashed')  

    # # Display the labels
    plt.xlabel('x positions (bins)')
    plt.ylabel('y positions  (counts)')
    plt.title('Pixel Histogram of Normalised Image')
    overlaid_histo_Img=root+'_overlaid_hist.png'
    plt.savefig(overlaid_histo_Img)
    return

def check_valid_np_type(raw_dtype:str,des_bg:float,des_fg:float):
    '''
    Function to check given raw data type is a valid int or float numpy type and then
    return the aproriatre max, min, air and materail values for that type.
    currently accpeted types are:
    np.float16, np.float32, np.int8, np.int16, np.int32,
    np.uint8, np.uint16, np.uint32.

    '''
    dt = np.dtype(raw_dtype)

    if dt.name in ['float16', 'float32']:
        dtype_min = 0.0
        dtype_max = 1.0
        pix_air = des_bg
        pix_material = des_fg
    elif dt.name in ['unit8','uint16', 'uint32']:
        dtype_max = np.iinfo(rawtype).max
        dtype_min = np.iinfo(rawtype).min
        pix_air = int(dtype_max*des_bg)
        pix_material = int(dtype_min*des_fg)
    else:
        raise ValueError(f'raw_type {dt.name} is not a valid int or float numpy type.')

    return dtype_min, dtype_max, pix_air, pix_material
