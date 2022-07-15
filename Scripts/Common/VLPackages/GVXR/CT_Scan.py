#!/usr/bin/env python3
from pickletools import uint8
import gvxrPython3 as gvxr
import numpy as np
import math
import meshio
import os
from Scripts.Common.VLPackages.GVXR.GVXR_utils import *
import numexpr as ne

def CT_scan(mesh_file,output_file,Beam,Detector,Model,Material_file=None,Headless=False,num_projections = 180,angular_step=1,im_format=None):
    ''' Main run function for GVXR'''
    # Print the libraries' version
    print (gvxr.getVersionOfSimpleGVXR())
    print (gvxr.getVersionOfCoreGVXR())

    # Create an OpenGL context
    print("Create an OpenGL context")
    if Headless:
    #headless
        gvxr.createWindow(-1,0,"EGL");
    else:
    # or with window
        gvxr.createWindow();
        gvxr.setWindowSize(512, 512);
    #

    # Load the data
    print("Load the data");
    
    mesh = meshio.read(mesh_file)

    #extract np arrays of mesh data from meshio
    points = mesh.points
    #triangles = mesh.get_cells_type('triangle')
    tetra = mesh.get_cells_type('tetra')

    if (not np.any(tetra)):
            raise ValueError("Input file must contain Tetrahedron data")

        # extract dict of material names and integer tags
    try:
        all_mat_tags=mesh.cell_tags
    except AttributeError:
        all_mat_tags = {}

    if not all_mat_tags:
        print ("[WARN] No materials defined in input file so we assume the whole mesh is made of a single material.")
        mat_tag_dict={0:['Un-Defined']}
        mat_ids = np.zeros(np.shape(tetra),dtype = int) 
        tags = np.unique(mat_ids)
    else:
    # pull the dictionary containing material id's for each tetrahderon
    # and the np array of ints that label the materials.
        mat_ids = mesh.get_cell_data('cell_tags','tetra')
        tags = np.unique(mat_ids)
        if(np.any(mat_ids==0)):
            all_mat_tags[0]=['Un-Defined']
        mat_tag_dict = find_the_key(all_mat_tags, np.unique(mat_ids))
            

    if Material_file is None:
        Material_file = 'Materials.csv'

    Material_file = os.path.abspath(Material_file)
            
    if os.path.exists(Material_file):
        Material_list = Read_Material_File(Material_file,mat_tag_dict)
        if len(tags) != len(Material_list):
            raise ValueError( f"Error: The number of Materials read in from {Material_file} does not match the number of materials in {inputfile}.")
    else:
        Material_list = Generate_Material_File(Material_file,mat_tag_dict)

    meshes=[]
    for N in tags:
        nodes = np.where(mat_ids==N)
        nodes=nodes[0]
        #set first value outside loop
        mat_nodes=tetra[nodes[0],np.newaxis]
        for M in nodes[1:]:
            mat_nodes=np.vstack([mat_nodes,tetra[M]])
        meshes.append(mat_nodes)


    #define boundray box for mesh
    min_corner = np.array([np.min(points[:,0]), np.min(points[:,1]), np.min(points[:,2])])
    max_corner = np.array([np.max(points[:,0]), np.max(points[:,1]), np.max(points[:,2])])
    bbox_range=max_corner-min_corner

    for vertex_id in range(len(points)):
        points[vertex_id][0] -= min_corner[0] + bbox_range[0] / 2.0;
        points[vertex_id][1] -= min_corner[1] + bbox_range[1] / 2.0;
        points[vertex_id][2] -= min_corner[2] + bbox_range[2] / 2.0;



    # Set up the beam
    print("Set up the beam")
    #gvxr.setSourcePosition(15,-40.0, 12.5, "mm");
    gvxr.setSourcePosition(Beam.PosX,Beam.PosY, Beam.PosZ, Beam.Pos_units);
    if (Beam.Beam_Type == 'point'):
        gvxr.usePointSource();
    elif (Beam.Beam_Type == 'parallel'):
        gvxr.useParallelBeam();
    else:
        raise ValueError(f"Invalid beam type {Beam.Beam_Type}, must be either point or parallel")

    gvxr.resetBeamSpectrum()
    for energy, count in zip(Beam.Energy,Beam.Intensity):
        gvxr.addEnergyBinToSpectrum(energy, Beam.Energy_units, count);
    
    # Set up the detector
    print("Set up the detector");
    #gvxr.setDetectorPosition(15.0, 80.0, 12.5, "mm");
    gvxr.setDetectorPosition(Detector.PosX,Detector.PosY, Detector.PosZ, Detector.Pos_units);
    gvxr.setDetectorUpVector(0, 0, -1);
    gvxr.setDetectorNumberOfPixels(Detector.Pix_X, Detector.Pix_Y);
    gvxr.setDetectorPixelSize(Detector.Spacing_X, Detector.Spacing_Y, Detector.Spacing_units);

    for i,mesh in enumerate(meshes):
        mesh = tets2tri(mesh,points)
        label = str(tags[i]);
        Mesh_Name = Material_list[i][0]
    ### BLOCK #####
        gvxr.makeTriangularMesh(Mesh_Name,
        points.flatten(),
        mesh.flatten(),
        "m");
        # place mesh at the orgin then traslate it according to the defined ofset
        #gvxr.moveToCentre(Mesh_Name);
        gvxr.translateNode(Mesh_Name,Model.PosX,Model.PosY,Model.PosZ,Model.Pos_units)
        gvxr.setElement(Mesh_Name, Material_list[i][1]);
        if i==0:
            gvxr.addPolygonMeshAsOuterSurface(Mesh_Name)
        else:
            gvxr.addPolygonMeshAsInnerSurface(Mesh_Name)
    # set initial rotation
    # note GVXR uses OpenGL which perfoms rotations with object axes not global.
    # This makes rotaions around the gloabal axes very tricky.
    M = len(Material_list)
    total_rotation = np.zeros((3,M))
    for i,N in enumerate(Material_list):
            
            # Gloabal X-axis rotation:
            global_axis_vec = world_to_model_axis(total_rotation[:,i],global_axis=[1,0,0]) # caculate vector along global x-axis in object co-odinates
            gvxr.rotateNode(N[0], Model.rotation[0], global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]); # perfom x rotation axis
            total_rotation[0,i] += Model.rotation[0]# track total rotation
            # Gloabal Y-axis rotation:
            global_axis_vec = world_to_model_axis(total_rotation[:,i],global_axis=[0,1,0]) # caculate vector along global Y-axis in object co-odinates
            gvxr.rotateNode(N[0], Model.rotation[1], global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]); # perfom Y rotation axis
            total_rotation[1,i] += Model.rotation[1]# track total rotation
            # Global Z-axis Rotaion:
            global_axis_vec = world_to_model_axis(total_rotation[:,i],global_axis=[0,0,1]) # caculate vector along global Z-axis in object co-odinates
            gvxr.rotateNode(N[0], Model.rotation[2], global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]); # perfom Z rotation axis
            total_rotation[2,i] += Model.rotation[2]# track total rotation 
    
    
    # Update the 3D visualisation
    gvxr.displayScene();       
    # Compute an X-ray image
    print("Compute CT aquisition");

    projections = [];
    theta = [];

    # calculate the rotation vector in model co-ordiantes that points
    # along the global axis
    # this is needed to alow us to rotate around the global axis rather than the cad model axis.
    global_axis_vec = world_to_model_axis(total_rotation[:,1],global_axis=[0,0,1]) # caculate vector along global Z-axis in object co-odinates
    for i in range(num_projections):
        # Compute an X-ray image and add it to the list of projections
        projections.append(gvxr.computeXRayImage());

        # Update the 3D visualisation
        gvxr.displayScene();
        # Rotate the model by angular_step degrees
        for i,N in enumerate(Material_list):
            gvxr.rotateNode(N[0], angular_step, global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]);
            total_rotation[2,i]+=angular_step
        theta.append(i * angular_step * math.pi / 180);

    # Convert the projections as a Numpy array
    projections = np.array(projections,'uint32');
    #return projections
    write_image(output_file,projections,im_format=im_format);
    # Perform the flat-Field correction of raw data
    dark = np.zeros(projections.shape);

    # Retrieve the total energy
    energy_bins = gvxr.getEnergyBins(Beam.Energy_units);
    photon_count_per_bin = gvxr.getPhotonCountEnergyBins();

    total_energy = 0.0;
    for energy, count in zip(energy_bins, photon_count_per_bin):
        total_energy += energy * count;
    flat = np.ones(projections.shape) * total_energy;

    #projections = flat_field_normalize(projections,flat,dark).astype('uint32')
    
    # Calculate  -log(projections)  to linearize transmission tomography data
    projections = minus_log(projections).astype('uint32')
    output_file = output_file + '_inv'
    write_image(output_file,projections,im_format=im_format);
    

    # Display the 3D scene (no event loop)
    # Run an interactive loop
    # (can rotate the 3D scene and zoom-in)
    # Keys are:
    # Q/Escape: to quit the event loop (does not close the window)
    # B: display/hide the X-ray beam
    # W: display the polygon meshes in solid or wireframe
    # N: display the X-ray image in negative or positive
    # H: display/hide the X-ray detector
    if (not Headless):
        controls_msg = ('### GVXR Window Controls ###\n'
        'You are Running an interactive loop \n'
        'You can rotate the 3D scene and zoom-in with the mouse\n'
        'buttons and scroll wheel.\n'
        ' \n'
        'To continue either close the window or press Q/Esc \n'
        ' \n'
        'Useful Keys are:\n'
        'Q/Escape: to quit the event loop\n'
        'B: display/hide the X-ray beam\n'
        'W: display the polygon meshes in solid or wireframe\n'
        'N: display the X-ray image in negative or positive\n'
        'H: display/hide the X-ray detector\n')
        print(controls_msg)
        gvxr.renderLoop();
    return 

def minus_log(arr):
    """
    Computation of the minus log of a given array using
    numexpr to speed up caculation over numpy.

    Parameters
    ----------
    arr : ndarray
        3D stack of projections.

    Returns
    -------
    ndarray
        Minus-log of the input data.
    """

    out = ne.evaluate('-log(arr)')
    print(out)
    return out


def flat_field_normalize(arr, flat, dark, cutoff=None):
    """
    Normalize raw projection data using the flat and dark field projections.
    Agaion using numexpr to Speed up calculations over plain numpy.

    Parameters
    ----------
    arr : ndarray
        3D stack of projections.
    flat : ndarray
        3D flat field data.
    dark : ndarray
        3D dark field data.
    cutoff : float, optional
        Permitted maximum value for the normalized data.
    
    Returns
    -------
    ndarray
        Normalized 3D tomographic data.
    """
    
    l = np.float32(1e-6)
    flat = np.mean(flat, axis=0, dtype=np.float32)
    dark = np.mean(dark, axis=0, dtype=np.float32)
    #get range for normalization
    denom = ne.evaluate('flat-dark')
    #remove values less than threshold l to avoid divide by zero
    ne.evaluate('where(denom<l,l,denom)', out=denom)
    out = ne.evaluate('arr-dark')
    out = ne.evaluate('out/denom', truediv=True)

    if cutoff is not None:
        cutoff = np.float32(cutoff)
        out = ne.evaluate('where(out>cutoff,cutoff,out)')
    
    return out
