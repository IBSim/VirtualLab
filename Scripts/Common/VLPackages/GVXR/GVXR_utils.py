import os
import csv
import errno
import pandas as pd
import numpy as np
from Scripts.Common.VLTypes.GVXR import Xray_Beam

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
    D = points[extra[0]]
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

def tets2tri(tetra,points):
    import itertools
    import time
    start = time.time()
    tri = np.empty((0,3),'int32')
    for i,tets in enumerate(tetra):
        nodes = itertools.combinations(tets,3)
        for tri_ind in nodes:
            tri_ind = list(tri_ind)
            extra = list(set(tetra[i]).difference(tri_ind))
            tri_ind = correct_normal(tri_ind,extra,points)
            tri=np.append(tri,np.array(tri_ind,ndmin=2),axis=0)
    stop = time.time()
    print(f"Conversion took: {stop-start} seconds")

    return tri

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
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), Material_file)

    print("Reading Materials from " + Material_file)
    df = pd.read_csv(Material_file)

    for i,row in enumerate(df["Material"]):
        #checking the data that is being read in
        check_Materials(row)
    
    #Materials_list = list(df["Material"].values)
    Materials_list = zip(df["Region Name"],df["Material"])

    return list(Materials_list)

def check_Materials(Mat_value):
    """ Function to check the element name or number is one that GVXR recognises.
        This is done by passing the string to the gvxr.getElementAtomicNumber(string).
        This already has error handling functions so if the name is valid it will return
        the atomic number (which we are not actually using) and if not it will 
        throw an exception and print. "ERROR: Element (name:string) not found."
 """
    import gvxrPython3 as gvxr
    atomic_number = gvxr.getElementAtomicNumber(Mat_value)
    return

def find_the_key(dictionary:dict, target_keys:str):
    """Function to pull out the keys of a dictionary as a list."""
    return {key: dictionary[key] for key in target_keys}

def write_image(output_file:str,vox:np.double,im_format:str=None):
    from PIL import Image, ImageOps
    import tifffile as tf
    if (im_format):
        for I in range(0,np.shape(vox)[2]):
            im = Image.fromarray(vox[:,:,I])
            im = ImageOps.grayscale(im)
            im_output="{}_{}.{}".format(output_file,I,im_format)
            im.save(im_output)
    else:
        im_output="{}.tiff".format(output_file)
        tf.imwrite(im_output,vox,photometric='minisblack')

def InitSpectrum(Beam: Xray_Beam,Verbose: bool=True,Show_plot: bool=False):
    ''' Function to create x-ray beam from spectrum generated by spekpy.'''
    
    import spekpy as sp

    if not hasattr[Beam,'Tube_Voltage']:
        raise ValueError('When using Spekpy you must define a Tube Voltage')
    if not hasattr[Beam,'Tube_Angle']:
        raise ValueError('When using Spekpy you must define a Tube Angle')
    if Verbose:
        print("Generating Beam Spectrum:" )
        print("Tube Voltage (kV):", Beam.Tube_Voltage)
        print("Tube Angle (degrees):", Beam.Tube_Angle)

    s = sp.Spek(kvp=Beam.Tube_Voltage, th=Beam.Tube_Angle) # Generate a spectrum (80 kV, 12 degree tube angle)

    if hasattr[Beam,'Filters']:
        for beam_filter in Beam.Filters:
            filter_material = beam_filter[0]
            filter_thickness_in_mm = beam_filter[1]

        if Verbose:
            print("Applying ", filter_thickness_in_mm, "mm thick Filter of ", filter_material)
        s.filter(filter_material, filter_thickness_in_mm)



    #units = "keV"
    k, f = s.get_spectrum(edges=True) # Get the spectrum

    if Show_plot:
        import matplotlib.pyplot as plt # Import library for plotting
        plt.plot(k, f) # Plot the spectrum",
        plt.xlabel('Energy [keV]')
        plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
        plt.show()

    spectrum={}
    for energy, count in zip(k, f):
        count = round(count)
        if count > 0:
            max_energy = max(max_energy, energy)
            min_energy = min(min_energy, energy)
            if energy in spectrum.keys():
                spectrum[energy] += count
            else:
                spectrum[energy] = count

    if Verbose:
        print("Minimum Energy:", min_energy, "keV")
        print("Maximum Energy:", max_energy, "keV")
    
    if hasattr[Beam,'Energy'] or hasattr[Beam,'Intensity']:
        import warnings
        warnings.warn('You have defined Energies or Intensity'
         'whist also using Spekpy. These will be ignored and'
         'replaced with the Spekpy Values.') 
    Beam.Energy = spectrum.keys()
    Beam.Intensity = spectrum.values()
    Beam.Energy_units = 'KeV'
    return Beam;