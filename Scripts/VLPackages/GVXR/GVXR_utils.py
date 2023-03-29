import os
import csv
import errno
import numpy as np
from operator import xor as xor_
from functools import reduce
import itertools
import time

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

def Check_Materials(Mat_list):
    """ Function to check the element name or number is one that GVXR recognises.
        This is done by passing the string to the gvxr.getElementAtomicNumber(string).
        This already has error handling functions so if the name is valid it will return
        the atomic number (which we are not actually using) and if not it will 
        throw an exception and print. "ERROR: Element (name:string) not found."
 """
    from gvxrPython3 import gvxr
    #import gvxrPython3 as gvxr
    for Mat_value in Mat_list:
        atomic_number = gvxr.getElementAtomicNumber(Mat_value)
    return

def find_the_key(dictionary:dict, target_keys:str):
    """Function to pull out the keys of a dictionary as a list."""
    return {key: dictionary[key] for key in target_keys}

def write_image(output_dir:str,vox:np.double,im_format:str='tiff',bitrate=8,angle_index=0):
    from PIL import Image, ImageOps
    import os
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    #calcualte number of digits in max number of images for formating
    import math
    if angle_index > 0:
        digits = int(math.log10(angle_index))+1
    elif angle_index == 0:
        digits = 1
    else:
        raise ValueError('Angle_index for write image must be a non negative int')

    if bitrate == 8:
        vox *= 255.0/vox.max()
        convert_opt='L'
    elif bitrate == 16:
        vox *= 65536/vox.max()
        convert_opt='I;16'
    elif bitrate == 32:
        convert_opt='F'
    else:
        print("warning: bitrate not recognised assuming 8-bit greyscale")
        convert_opt='L'

    im = Image.fromarray(vox)
    im = im.convert(convert_opt)
    im_output=f"{output_dir}/{output_name}_{angle_index:0{digits}d}.{im_format}"
    im.save(im_output)

def write_image3D(output_dir:str,vox:np.double,im_format:str='tiff',bitrate=8):
    from PIL import Image, ImageOps
    import os
    output_name = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    #calcualte number of digits in max number of images for formating
    import math
    digits = int(math.log10(np.shape(vox)[0]))+1
    if bitrate == 8:
        vox *= 255.0/vox.max()
        convert_opt='L'
    elif bitrate == 16:
        vox *= 65536/vox.max()
        convert_opt='I;16'
    elif bitrate == 32:
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


# def InitSpectrum(Beam,Headless:bool=False):
#     ''' Function to create x-ray beam from spectrum generated by spekpy.
#     ====================================================================
#     Input Pratmeters:
#     -----------------
#     Beam: Xray_Beam - dataclass to hold data related to xray beam.
#     Headless: bool - flag to set if we want turn on/off matplot lib plots of spectrum data.
#     '''
    
#     import spekpy as sp
#     if (Beam.Tube_Voltage == 0.0):
#         raise GVXRError('When using Spekpy you must define a Tube Voltage')

#     kwargs = {'kvp':Beam.Tube_Voltage,'th':Beam.Tube_Angle}

#     print("Generating Beam Spectrum:" )
#     print("Tube Voltage (kV):", Beam.Tube_Voltage)
#     print("Tube Angle (degrees):", Beam.Tube_Angle)

#     if Beam.Tube_Voltage > 300.0:
#         print(f"Warning: Beam Tube voltage exceeds 300Kv which is the max that spekpy supports. \n \
#         Thus beam enegry has been set to a flat {Beam.Tube_Voltage} keV.")
#         Beam.Energy = [Beam.Tube_Voltage]
#         Beam.Intensity = [1000]
#         Beam.Energy_units = 'keV'
#         return Beam;
#     s = sp.Spek(**kwargs) # Generate a spectrum

#     if xor(Beam.Filter_ThicknessMM==None,Beam.Filter_Material==None):
#         # only one of the two parameters has been set
#         raise GVXRError(f'When using Spekpy with a filter you must define both Filter_Thickness and Material:\n \
#         Filter_Thickness={Beam.Filter_ThicknessMM} \n \
#         Filter_Material={Beam.Filter_Material}')

#     elif (Beam.Filter_ThicknessMM!=None) and (Beam.Filter_Material!=None):
#         # both have been correctly set so add filtering
#         print(f"Applying {Beam.Filter_ThicknessMM} mm thick Filter of {Beam.Filter_Material}")
#         s.filter(str(Beam.Filter_Material), float(Beam.Filter_ThicknessMM))
#     else:
#         # Neither have been set so do nothing.
#         print("No Beam Filtering was applied")


#     #units = "keV"
#     k, f = s.get_spectrum(edges=True) # Get the spectrum
#     if not Headless:
#         import matplotlib.pyplot as plt # Import library for plotting
#         plt.plot(k, f) # Plot the spectrum",
#         plt.xlabel('Energy [keV]')
#         plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
#         plt.show()

#     max_energy = 0
#     min_energy = 0
#     spectrum={}
#     for energy, count in zip(k, f):
#         count = round(count)
#         if count > 0:
#             max_energy = max(max_energy, energy)
#             min_energy = min(min_energy, energy)
#             if energy in spectrum.keys():
#                 spectrum[energy] += count
#             else:
#                 spectrum[energy] = count
    
#     if (Beam.Energy is not None) or (Beam.Intensity is not None):
#         import warnings
#         warnings.warn('You have defined Energies or Intensity'
#          'whist also using Spekpy. These will be ignored and'
#          'replaced with the Spekpy Values.') 
#     Beam.Energy = list(spectrum.keys())
#     Beam.Intensity = list(spectrum.values())
#     Beam.Energy_units = 'keV'
#     return Beam;

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