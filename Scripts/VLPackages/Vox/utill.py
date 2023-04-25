"""A bunch of useful utility functions"""
import numpy as np

def convert_units(values,input_units='mm',output_units='mm'):
    '''
    Function to convert unit_length from Vox_units into mesh_units
    units must be one of 'mm','cm' or 'm'.
    '''
    unit_list = ['mm','cm','m']
    if input_units.lower() not in unit_list:
        raise ValueError(f'Invalid Voxel units {input_units} must be one of {unit_list}')
    if output_units.lower() not in unit_list:
        raise ValueError(f'Invalid Mesh units {output_units} must be one of {unit_list}')

    if input_units == 'mm':
        numerator = 1.0
    elif input_units == 'cm':
        numerator = 10.0
    else:
        numerator = 1000.0

    if output_units == 'mm':
        denominator = 1.0
    elif output_units == 'cm':
        denominator = 10.0
    else:
        denominator = 1000.0
    
    output=[0.0,0.0,0.0]
    for I,_ in enumerate(values):
        output[I] = values[I] * (numerator/denominator)
    return output
    
# check if greyscale can be converted into a vaild 8-bit int
def check_greyscale(greyscale_value):
    """Function to check the user defined Greyscale values are valid 8-bit integers"""
    try:
        int(greyscale_value)
    except ValueError:
        print(greyscale_value)
        raise ValueError(
            "Invalid Greyscale value, must be an Integer value, "
            "or castable to and Integer value"
        )

    if (int(greyscale_value) < 0) or (int(greyscale_value) > 255):
        raise ValueError("Invalid Greyscale value. Must be between 0 and 255")


def find_the_key(dictionary, target_keys):
    """Function to pull out the keys of a dictionary as a list."""
    return {key: dictionary[key] for key in target_keys}


def check_gridsize(gridsize):
    """check that gridsize is a list of 3 non-zero positive integers"""
    if not isinstance(gridsize, list):
        raise TypeError("Invalid Gridsize. Must be a list.")
    if len(gridsize) != 3:
        raise TypeError("Invalid Gridsize. Must be a list of three integer values.")

    for i in gridsize:
        if not isinstance(i, int):
            raise TypeError(f"Invalid Gridsize {i}. Must be an integer value.")
        if i < 0:
            raise TypeError(
                f"Invalid Gridsize {i}. Must be an integer value that is"
                "greater than 0."
            )


def check_padding(Output_Resolution):
    """
    Check that Output_Resolution is a list of 3 non-zero positive integers.
    """
    if not isinstance(Output_Resolution, list):
        raise TypeError("Invalid Output_Resolution. Must be a list.")
    if len(Output_Resolution) != 3:
        raise TypeError(
            "Invalid Output_Resolution. Must be a list of three integer values."
        )

    for i, j in enumerate(Output_Resolution):
        if not isinstance(j, int):
            raise TypeError(f"Invalid Output_Resolution {j}. Must be an integer value.")
        if i < 0:
            raise TypeError(
                f"Invalid Output_Resolution {j}. Must be an integer value that is"
                "greater than 0."
            )


def check_unit_length(unit_length,mesh_units='mm',Voxel_units='mm'):
    """check that unit_length is a list of three non-zero positive floats."""
    if not isinstance(unit_length, list):
        raise TypeError("Invalid unit_length. Must be a list.")
    if len(unit_length) != 3:
        raise TypeError("Invalid unit_length. Must be a list of three floats.")

    for i in unit_length:
        if not isinstance(i, float):
            raise TypeError(f"Invalid unit length {i} Must be a floating point value.")
        if i < 0:
            raise TypeError(
                f"Invalid unit length {i}. Must be a floating point value"
                " that is greater than 0."
            )
    unit_length = convert_units(unit_length,Voxel_units,mesh_units)

def check_voxinfo(
    unit_length=[0.0, 0.0, 0.0], gridsize=[0, 0, 0], mesh_min=None, mesh_max=None,
    mesh_units='mm',Voxel_units='mm'
):
    """

    check to see that both unit_length and gridsize are valid and at least one is defined.
    After which it calculates the size of the image boundary box.



    Note: the calculation of unit_length is in reality handled deep in the C++
    code. This is because the code incorporates a small displacement to the grid
    "epsilon" to try and avoid vertices falling directly on the grid
    (see "util.h" line 98). Thus this function is a bit of a hack in that it
    although we calculate the unit_length we never actually use it.

    It just hands gridsize off to the c++ code to calculate the "actual" unit_length.

    Note this function converts gridsize from a list to an np array in the final step.

    """
    import numpy as np
    import math

    #  check gridsize and unit_length are valid.
    check_gridsize(gridsize)
    check_unit_length(unit_length,mesh_units,Voxel_units)

    if (gridsize == [0, 0, 0]) and (0.0 not in unit_length):
        # unit_length has been defined by user so check it is valid and
        # then calculate gridsize.
        for i, _ in enumerate(gridsize):
            gridsize[i] = int((math.ceil((mesh_max[i] - mesh_min[i]) / unit_length[i])))
            Bbox_min = mesh_min
            Bbox_max = mesh_max
    # GridSize has been defined by user so check it is valid and
    # then calculate unit_length.
    elif (unit_length == [0.0, 0.0, 0.0]) and (0 not in gridsize):
        for i, _ in enumerate(gridsize):
            unit_length[i] = float((mesh_max[i] - mesh_min[i]) / gridsize[i])
        Bbox_min = mesh_min
        Bbox_max = mesh_max

    elif (gridsize == [0, 0, 0]) and (unit_length == [0.0, 0.0, 0.0]):
        # Neither has been defined
        raise TypeError("You must define one of either Gridsize or unit_length")

    else:
        # Both have been defined by user in which case we calculate the image boundary's
        raise TypeError("You must define only one of either Gridsize or unit_length.")

    gridsize = np.array(gridsize)
    Vox_info = {"gridsize": gridsize, "Bbox_min": Bbox_min, "Bbox_max": Bbox_max}
    return Vox_info


def crop_center(img, outresx, outresy, outresz, rm_start=True):
    """
    Take a 3D array and crop it from the centre in 3D.
    outresx, outresy, outresz represents your desired dims in x,y and z.

    Note if these are smaller than the corresponding dimension of img that 
    dimension will be unchanged.

    Also Note: ideally we want to remove an equal number of values from the start and 
    end of each dimension. However this is obviously impossible if the difference 
    between the current and desired dimension is not even. In which case we have 1 
    additional value to remove. Thus we have provided an extra param rm_start.
    
    This is a bool to control where to remove the "extra" value from. If True (the default)
    it removes the it from the start of the dim. If False we remove it from the end.

    """
    x, y, z = img.shape

# don't crop anything if output res is greater than the image current resolution in that dim.
    if outresx > x :
        cropx = 0
        remx = 0
    else:
    # otherwise drop an equal number of values from each side
        cropx = (x - outresx)//2
        if  (x - outresx) % 2 == 0:
            remx = 0
        else:
        # difference is odd so we need to remove an additional value from 
        # either the right of left hand side. 
            remx = 1
        
    if outresy > y:
        cropy = 0
        remy = 0
    else:
        cropy = (y - outresy)//2
        if  (y - outresy) % 2 == 0:
            remy = 0
        else:
            remy = 1

    if outresz > z:
        cropz = 0
        remz = 0
    else:
        cropz = (z - outresz)//2
        if  (z - outresz) % 2 == 0:
            remz = 0
        else:
            remz = 1
    
    startx = cropx
    stopx = x - cropx
    starty = cropy
    stopy = y - cropy
    startz = cropz
    stopz = z - cropz

    if rm_start:    
        # remove additional value from the start
        startx = startx + remx
        starty = starty + remy
        startz = startz + remz
    else:
    # remove additional value from the end
        stopx = stopx - remx
        stopy = stopy - remy
        stopz = stopz - remz

    return img[startx:stopx, starty:stopy, startz:stopz]


def padding(array, xx, yy, zz):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :param zz: desired length
    :return: padded array
    """

    h, w, l = array.shape
    if h < xx:
        a = (xx - h) // 2
        aa = xx - a - h
    else:
        a = 0
        aa =0
    if w < yy:    
        b = (yy - w) // 2
        bb = yy - b - w
    else:
        b = 0
        bb =0
    if l < zz:
        c = (zz - l) // 2
        cc = zz - c - l
    else:
        c = 0
        cc = 0

    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode="constant")
