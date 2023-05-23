import os
import csv
import errno
import numpy as np
from operator import xor as xor_
from functools import reduce
import itertools
import time
import VLconfig as VLC
import math

class GVXRError(Exception):
    '''Custom error class to format error message in a pretty way.'''
    def __init__(self, value): 
        self.value = value
    def __str__(self):
        Errmsg = "\n========= Error =========\n\n"\
        f"{self.value}\n\n"\
        "=========================\n\n"
        return Errmsg

def warning_message(msg:str,line_length:int=50):
    ''' 
    Function to take in a long string and format it in a pretty
    way for printing to screen.
    '''
    from textwrap import fill
    header = " WARNING: ".center(line_length,'*')
    footer = ''.center(line_length,'*')
    msg = fill(msg,width=line_length,initial_indent=' ', subsequent_indent=' ')
    print(header)
    print(msg)
    print(footer)
    return


def xor(*args):
    return reduce(xor_, map(bool, args))

def FlipNormal(triangle_index_set):
    ''' Function to swap index order of triangles to flip the surface normal.'''
    triangle_index_set = np.array(triangle_index_set)
    triangle_index_set[[0,1,2]] = triangle_index_set[[0,2,1]]
    return triangle_index_set

def correct_normal(tri_ind,extra,points):
    ''' Function to calcuate the dot product between the surface normal of a 
    traingle defined by points A, B, C and the vector between C and a point D.
    This is used to check the direction of the surface normal for each face of
    a tetrahedron. If the Normal points outwards, as it should the dotproduct 
    will be postive. However, If the Dp is -ve then the normal is pointing 
    the wrong way (inwards) so we need to shuffle the points in the triangle
    to flip it.'''
    A = points[tri_ind[0]]
    B = points[tri_ind[1]]
    C = points[tri_ind[2]]
    D = points[extra]
    #edge vectors 
    E0 = B - A
    E1 = C - B
    #test vector
    AD = D - A
    Surf_Norm = np.cross(E0,E1)
    test = float(np.dot(Surf_Norm,AD))
    if test > 0:
        tri_ind = FlipNormal(tri_ind)
    return tri_ind

def extract_unique_triangles(t):
    sort_tri = np.sort(t,axis=1)
    _,index,counts = np.unique(sort_tri,axis=0,return_index=True, return_counts=True)
    tri = t[index[counts==1]]
    return tri, index[counts==1]

def fix_normals_full(tet,points):
    '''
    Function to check and fix if necessary the 4 surface normals of a given tetrahedron
    Note we use functools.partail when calling this function to prevent us having to 
    copy the points array for each call given it never changes.
    '''
    Nodes = list(itertools.combinations(tet,3))
    surface= np.empty([len(Nodes),3])
    for I, face in enumerate(Nodes):
        face = np.array(face)
        extra = int(np.setdiff1d(tet,face,assume_unique=True))
        face = correct_normal(face,extra,points)
        surface[I,:] = np.array([face[0],face[1],face[2]])
    return surface

def tets2tri(tetra,points,mat_ids):
    import functools
    import multiprocessing
    start = time.time()
    # each tet has been broken dwon into 4 triangles 
    # so we must expand mat_ids by 4 times to get the
    #  id for each tri.
    #mat_ids = np.repeat(mat_ids,4)
    vol_mat_ids = np.empty(len(tetra)*4,'int')
    surface = []
    # tri=np.empty(3,dtype='int32')
    surf_mat_ids=[]
    items=[]
    j = 0
    # we use functools.partail here to prevent us having to copy the points array for each call to fix_normals
    fix_normals = functools.partial(fix_normals_full,points=points)

    with multiprocessing.Pool() as pool:
	# call the function for each item in parallel
	    for result in pool.map(fix_normals, tetra):
               surface.append(result)
    tri = np.concatenate(surface,axis=0).astype('int32')
    vol_mat_ids = np.repeat(mat_ids,4)     
    # extract triangles on the surface of the mesh and there id's
    tri_surf, ind = extract_unique_triangles(tri)
    
    surf_mat_ids = np.take(vol_mat_ids,ind)
    stop = time.time()
    print(f"Tet to Tri conversion took: {stop-start} seconds")

    return tri_surf, surf_mat_ids

def Generate_Material_File(material_file,mat_tags):
    """ Function to generate a new Materail file if none are defined.
    This file is populated with names/indicies read in from mesh and
     each region given the default material of Copper"""
    # create list of tags and values for each mesh region.
    mat_index = list(mat_tags.keys())
    mat_names = list(mat_tags.values())
    
    print("writing Mesh Materails to " + material_file)
    with open(material_file, 'w',encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Region Name","index","Material"])
        mat_list=[]
        for i,names in enumerate(mat_names):
            row = [" ".join(names),mat_index[i],"Cu"]
            writer.writerow(row)
            mat_list.append((names,"Cu"))

    return mat_list

def Read_Material_File(Material_file,mat_tags):
    """ Function to Read Material values from file if a file is defined by the user."""
    Material_file = os.path.abspath(Material_file)
    num_mats = len(mat_tags)
    if not os.path.exists(Material_file):
        raise GVXRError(errno.ENOENT, os.strerror(errno.ENOENT), Material_file)

    print("Reading Materials from " + Material_file)
    df = pd.read_csv(Material_file)

    
    
    #Materials_list = list(df["Material"].values)
    Materials_list = zip(df["Region Name"],df["Material"])

    return list(Materials_list)

def Check_Materials(Mat_list,Amounts,Density):
    """ 
    Function to check the element name or number is one that GVXR recognizes.
    Then create a list of strings for each materials corresponding type with 
    E for element, M for mixture, and C for compound. This will then tell 
    GVXR what to do with each material.
    """
    Mixture = True
    mat_types = []        
    for Material in Mat_list:
    # Check mat_list for list of lists, This indicates that the user has defined a mixture
    # as such we then need to check they have defined both amounts and Density
    # otherwise Assume they have defined either elements or compounds
        if isinstance(Material, list):
            if not all(isinstance(x, int) for x in Material):
                raise ValueError(f'Invalid mixture {Material}. When using a mixture of elements they must be all be ints.')
            elif Amounts == []:
                raise ValueError(f'When using a mixture of elements you must define but not GVXR.Amounts')
            elif len(Mat_list) != len(Amounts):
               raise ValueError(f'You have defined {len(Mat_list)} mixtures but {len(Amounts)} corresponding amounts. These must match')
            else:
                mat_types.append('M')

        elif isinstance(Material, int):
            check_element_num(Material)
            mat_types.append('E')
        elif isinstance(Material, str):
            mtype = check_element_or_compound(Material)
            mat_types.append(mtype)
        else:
           raise ValueError(f'Invalid material {Materail} of type {type(Materail)} must be an int or a string.')

    # if mixture or compound have been used we need to check that the Density was defined for each
    if any(s in mat_types for s in ('M','C')):
        if Density == []:
            raise ValueError(f'When using a mixture of elements or Compounds you must define GVXR.Density')
        elif len(Mat_list) != len(Density):
            raise ValueError(f'You have defined {len(Mat_list)} mixtures/compunds but {len(Density)} corresponding Densities. These must match')
    
    return mat_types

def check_element_num(number:int):
    '''.
    Simple function to check if given element number is valid
    Valid elements are ints between 1 and 100.  
    '''
    if  number < 0:
        raise ValueError(f'Invalid Atomic number {number} must not be negative.')
    elif number > 100:
        raise ValueError(f'Invalid Atomic number {number} must be less than 100.')

    return

def check_element_or_compound(Material:str):
    '''
    function to check if string is a element name, symbol or a compound.
    Note: we have no good way of checking if the string is a valid compound.
    As such we assume that if it is not an element name or symbol it must 
    be a compound. in which case we are relying on GVXR to check if its valid.  
    '''
    import csv

#  Open csv file containing names and symbols for elements 1 to 100 
    csv_file = open(f'{VLC.VL_DIR_CONT}/Scripts/VLPackages/GVXR/ptable.csv','r')
    z_num = []
    element_names = []
    symbols = []

    # Read off and discard first line, to skip headers
    csv_file.readline()

# Split columns while reading
    for a, b, c in csv.reader(csv_file, delimiter=','):
    # Append each variable to a separate list
        z_num.append(a)
        element_names.append(b)
        symbols.append(c)
    
    if Material in element_names or Material in symbols:
        mtype = 'E'
    else:
        mtype = 'C'

    return mtype

def find_the_key(dictionary:dict, target_keys:str):
    """Function to pull out the keys of a dictionary as a list."""
    return {key: dictionary[key] for key in target_keys}

def write_image(output_dir:str,vox:np.double,im_format:str='tiff',bitrate='float32',angle_index=0):
    from PIL import Image, ImageOps
    import os
    import tifffile
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if im_format == None:
        im_format:str='tiff'
    #calcualte number of digits in max number of images for formating
    import math
    if angle_index > 0:
        digits = int(math.log10(angle_index))+1
    elif angle_index == 0:
        digits = 1
    else:
        raise ValueError('Angle_index for write image must be a non negative int')

    if bitrate == 'int8':
        vox *= 255.0/vox.max()
        convert_opt='L'
    elif bitrate == 'int16':
        vox *= 65536/vox.max()
        convert_opt='I;16'
    elif bitrate == 'float32':
        convert_opt='F'
    else:
        print(f"warning: bitrate {bitrate} not recognized assuming 8-bit grayscale")
        convert_opt='L'

    im = Image.fromarray(vox)
    im = im.convert(convert_opt)
    im_output=f"{output_dir}/{output_name}_{angle_index:0{digits}d}.{im_format}"
    im.save(im_output)
    im.close()


def write_image3D(output_dir:str,vox:np.double,im_format:str='tiff',bitrate='float32'):
    from PIL import Image, ImageOps
    import os
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    #calcualte number of digits in max number of images for formating
    import math
    digits = int(math.log10(np.shape(vox)[0]))+1
    if bitrate == 'int8':
        vox *= 255.0/vox.max()
        convert_opt='L'
    elif bitrate == 'int16':
        vox *= 65536/vox.max()
        convert_opt='I;16'
    elif bitrate == 'float32':
        convert_opt='F'
    else:
        print("warning: bitrate not recognised assuming 8-bit greyscale")
        vox *= 255.0/vox.max()
        convert_opt='L'

    for I in range(0,np.shape(vox)[0]):
        im = Image.fromarray(vox[I,:,:])
        im = im.convert(convert_opt)
        im_output=f"{output_dir}/{output_name}_{I:0{digits}d}.{im_format}"
        im.save(im_output)
        im.close()

def without_keys(d, keys):
    return {x: d[x] for x in d if x not in keys}

def dump_to_json(Python_dict:dict,file_name:str):
    import json
    import dataclasses as dc
    # Remove all parmas that are Dataclasses as we will deal with them seperatly
    IO_params = without_keys(Python_dict,["Beam","Detector","Model"])
    #extract datclasses as seperate dicts
    Beam = Python_dict["Beam"].__dict__
    Cad = Python_dict["Model"].__dict__
    Det = Python_dict["Detector"].__dict__

    params ={**IO_params, **Beam, **Cad,**Det}
    with open(file_name, 'w') as fp:
        json.dump(params, fp)
        fp.close()

def world_to_model_axis(rotation,global_axis=[0,0,1],threshold=1E-5):
    '''because Rotations in openGL are based around object axes not
     global axes we need to calculate the unit vector that points 
     along the global x,y or z axis in the model co-ordinates order 
     to rotate around it for the CT scan.

     inputs: 
     Rotation - How many degrees have you rotated the model around the x,y and z axis
     global_axis - unit vector pointing along the global axis you wish to rotate around.
     
     Outputs:
     model_axis - unit vector that points along the specified world axis in the 
     objects co-odinate system. '''
    from scipy.spatial.transform import Rotation as R
# calculate euler representaion of rotation
    r = R.from_euler('xyz',rotation,degrees=True)
    r_inv = r.inv()
# apply rotaion to axis
    model_axis = r_inv.apply(global_axis)
    return model_axis

def convert_tets_to_tri(mesh_file):
    '''
    Function to read in a tetrahedron based 
    volume mesh with meshio and convert it 
    into surface triangle mesh for use with 
    GVXR.
    '''
    import numpy as np
    import meshio
    import os
    root, ext = os.path.splitext(mesh_file)
    new_mesh_file = f"{root}_triangles{ext}"
    # This check helps us avoid having to repeat the conversion from tri to tet 
    # for reach run when using one mesh file for multiple GVXR runs.
    if os.path.exists(new_mesh_file):
        print(f"Found {new_mesh_file} so assuming conversion has already been done previously.")
        return new_mesh_file
    
    print("Converting tetrahedron mesh into triangles for GVXR")
    mesh = meshio.read(mesh_file)
    #extract np arrays of mesh data from meshio
    points = mesh.points
    tetra = mesh.get_cells_type('tetra')
    if not np.any(tetra):
        #no tetra data but trying to use tets
        raise ValueError("User asked to use tets but mesh file does not contain Tetrahedron data")
    mat_ids_tet = mesh.get_cell_data('cell_tags','tetra')
    #extract surface triangles from volume tetrahedron mesh
    elements, mat_ids  = tets2tri(tetra,points,mat_ids_tet)
    cells = [('triangle',elements)]

    # convert extracted triangles into new meshio object and write out to file
    tri_mesh = meshio.Mesh(
        points,
        cells,
        # Each item in cell data must match the cells array
        cell_data={"cell_tags":[mat_ids]},
    )
    tri_mesh.cell_tags = find_the_key(mesh.cell_tags, np.unique(mat_ids))
    print(f"Saving new triangle mesh as {new_mesh_file}")
    tri_mesh.write(new_mesh_file)

    return new_mesh_file

def fill_edges(projection:'np.ndarray[np.float32]',fill_percent:float,fill_value:float=None, nbins:int = 256):
    '''
    Function to fill in the edges of an xray projection with fill_value (default is 1.0 i.e. background).
    This is done to reduce ring/halo artifacts during reconstruction with CIL.
    param: projection - 2D numpy array representing the image
    param:  fill_percent -  float representing the percentage of pixes y ou want to fill 
                            in from each edge.
    param: fill_value - value to replace pixels with, default is to calcuate background value using image histogram.
    '''
    if fill_percent == None or fill_percent == 0.0:
        return projection
    
    pix_y,pix_x = np.shape(projection)

    # calculate the number of pixes to remove from each side in x and y
    # Note: we use floor here since we dont want any chance that nx or ny
    # are bigger then half of pix_x or pix_y.
    nx = int(math.floor((pix_x * fill_percent)/ 2))
    ny = int(math.floor((pix_y * fill_percent)/ 2))
    if fill_value == None:
        # use a histgram of the image to pick out the highest peak to obtain background intensity
        bins, vals = np.histogram(projection,nbins)
        peak_ind = np.argmax(bins)
        fill_value = vals[peak_ind]
    projection[:ny,:] = fill_value
    projection[:,:nx] = fill_value
    # this stops you nuking every value in the array instead of filling 0 
    # pixels because we are using -ve indexing and as such -0 is treat as 0.
    if ny > 0:
        projection[-ny:,:] = fill_value 
    if nx > 0:
        projection[:,-nx:] = fill_value

    return projection

