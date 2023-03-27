#!/usr/bin/env python3
from pickletools import uint8
from gvxrPython3 import gvxr
from gvxrPython3.utils import loadXpecgenSpectrum
#import gvxrPython3 as gvxr
import numpy as np
import math
import meshio
from Scripts.VLPackages.GVXR.GVXR_utils import *

def CT_scan(**kwargs):
    ''' Main run function for GVXR'''
    #get kwargs or set defaults
    Material_list = kwargs['Material_list']
    Headless = kwargs.get('Headless',False)
    num_projections = kwargs.get('num_projections',180)
    angular_step = kwargs.get('angular_step',1)
    im_format = kwargs.get('im_format','tiff')
    use_tetra = kwargs.get('use_tetra',False)
    downscale = kwargs.get('downscale',1.0)
    print(gvxr.getVersionOfSimpleGVXR())
    print(gvxr.getVersionOfCoreGVXR())
    # Create an OpenGL context
    print("Create an OpenGL context")
    if Headless:
    # headless
        gvxr.createWindow(-1, 0, "EGL", 4, 5)
    else:
        gvxr.createWindow(-1, 1, "OPENGL", 4, 5)

    # Load the data
    print("Loading the data");

    if use_tetra:
       mesh_file = convert_tets_to_tri(kwargs['mesh_file'])
       mesh = meshio.read(mesh_file)
    else:
        mesh = meshio.read(kwargs['mesh_file'])

    #extract np arrays of mesh data from meshio
    points = mesh.points
    triangles = mesh.get_cells_type('triangle')
    tetra = mesh.get_cells_type('tetra')

    if (not np.any(triangles) and not np.any(tetra)):
        raise GVXRError("Input file must contain one of either Tets or Triangles")

    if not np.any(triangles) and not use_tetra:
        #no triangle data but trying to use triangles
        raise GVXRError("User asked to use triangles but input file does "
        "not contain Triangle data")
        
        # extract dict of material names and integer tags
    try:
        all_mat_tags=mesh.cell_tags
    except AttributeError:
        all_mat_tags = {}

    if all_mat_tags == {}:
        print ("[WARN] No materials defined in input file so we assume the whole mesh is made of a single material.")
        mat_tag_dict={0:['Un-Defined']}
        all_mat_tags = mat_tag_dict
        mat_ids = np.zeros(np.shape(triangles)[0],dtype = int) 
        tags = np.unique(mat_ids)
    else:
    # pull the dictionary containing material id's for each element
    # and the np array of ints that label the materials.
        mat_ids = mesh.get_cell_data('cell_tags','triangle')
        
        tags = np.unique(mat_ids)
        if(np.any(mat_ids==0)):
            all_mat_tags['0']=['Un-Defined']
        mat_tag_dict = find_the_key(all_mat_tags, np.unique(mat_ids))
            
    elements = triangles

    if len(tags) != len(Material_list):
        Errormsg = (f"Error: The number of Materials read in from Input file is {len(Material_list)} "
        f"this does not match \nthe {len(mat_tag_dict)} materials in {kwargs['mesh_file']}.\n\n" 
        f"The meshfile contains: \n {mat_tag_dict} \n\n The Input file contains:\n {Material_list}.")
        gvxr.destroyAllWindows();
        raise GVXRError(Errormsg)

    meshes=[]
    mesh_names=[]
    for N in tags:
        nodes = np.where(mat_ids==N)
        nodes=nodes[0]
        mat_nodes = np.take(elements,nodes,axis=0)
        meshes.append(mat_nodes)
        mesh_names.append(str(all_mat_tags[N]))
    

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
    gvxr.setSourcePosition(kwargs['Beam_PosX'],kwargs['Beam_PosY'], kwargs['Beam_PosZ'], kwargs['Beam_Pos_units']);
    if (kwargs['Beam_Type'] == 'point'):
        gvxr.usePointSource();
    elif (kwargs['Beam_Type'] == 'parallel'):
        gvxr.useParallelBeam();
    else:
        raise GVXRError(f"Invalid beam type {kwargs['Beam_Type']} defined in Input File, must be either point or parallel")

    gvxr.resetBeamSpectrum()
    if kwargs["Tube_Voltage"] != 0.0:
        #generate an xray tube spectrum
        filters = []
        if kwargs["Filter_Material"] != None and kwargs["Filter_ThicknessMM"] != None:
            materiails = [kwargs["Filter_Material"]]
            thickness = [kwargs["Filter_ThicknessMM"]]
            for mat, thick in zip(materiails,thickness):
                filters.append([mat,thick])
        T_angle = kwargs.get("Tube_Angle",12.0)
        print(f"generating xray Tube spectrum for {kwargs['Tube_Voltage']} Kv tube.")
        spectrum_filtered, k_filtered, f_filtered, units = loadXpecgenSpectrum(kwargs["Tube_Voltage"],filters=filters)
    else:
    # generate spectrum from given energy and intensity values
        print("Generating Beam spectrum using supplied values of Energy and Intensity.")
        for energy, count in zip(kwargs['Energy'],kwargs['Intensity']):
            gvxr.addEnergyBinToSpectrum(energy, kwargs['Energy_units'], count);
    
    # Set up the detector
    print("Set up the detector");
    #gvxr.setDetectorPosition(15.0, 80.0, 12.5, "mm");
    gvxr.setDetectorPosition(kwargs['Det_PosX'],kwargs['Det_PosY'], kwargs['Det_PosZ'], kwargs['Det_Pos_units']);
    gvxr.setDetectorUpVector(0, 0, -1);
    gvxr.setDetectorNumberOfPixels(kwargs['Pix_X'], kwargs['Pix_Y']);
    gvxr.setDetectorPixelSize(kwargs['Spacing_X']*downscale, kwargs['Spacing_Y']*downscale, kwargs['Spacing_units']);
    for i,mesh in enumerate(meshes):
        label = mesh_names[i];
    ### BLOCK #####
        gvxr.makeTriangularMesh(label,points.flatten(),mesh.flatten(),str(kwargs['Model_Mesh_units']));
        # place mesh at the origin then translate it according to the defined offset
        gvxr.moveToCentre(label);
        gvxr.translateNode(label,kwargs['Model_PosX'],kwargs['Model_PosY'],kwargs['Model_PosZ'],kwargs['Model_Pos_units'])
        gvxr.scaleNode(label, kwargs['Model_ScaleX'], kwargs['Model_ScaleY'], kwargs['Model_ScaleZ'])
        gvxr.setElement(label, Material_list[i]);
        gvxr.addPolygonMeshAsInnerSurface(label)
        
    # set initial rotation
    # note GVXR uses OpenGL which performs rotations with object axes not global.
    # This makes rotations around the global axes very tricky.
    M = len(mesh_names)
    total_rotation = np.zeros((3,M))
    for i,label in enumerate(mesh_names):
            # Gloabal X-axis rotation:
            global_axis_vec = world_to_model_axis(total_rotation[:,i],global_axis=[1,0,0]) # caculate vector along global x-axis in object co-odinates
            gvxr.rotateNode(label, kwargs['rotation'][0], global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]); # perfom x rotation axis
            total_rotation[0,i] += kwargs['rotation'][0]# track total rotation
            # Gloabal Y-axis rotation:
            global_axis_vec = world_to_model_axis(total_rotation[:,i],global_axis=[0,1,0]) # caculate vector along global Y-axis in object co-odinates
            gvxr.rotateNode(label, kwargs['rotation'][1], global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]); # perfom Y rotation axis
            total_rotation[1,i] += kwargs['rotation'][1]# track total rotation
            # Global Z-axis Rotaion:
            global_axis_vec = world_to_model_axis(total_rotation[:,i],global_axis=[0,0,1]) # caculate vector along global Z-axis in object co-odinates
            gvxr.rotateNode(label, kwargs['rotation'][2], global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]); # perfom Z rotation axis
            total_rotation[2,i] += kwargs['rotation'][2]# track total rotation
    # Update the 3D visualisation
    gvxr.displayScene();       
    # Compute an X-ray image
    print("Compute CT aquisition");

    theta = [];
    # Compute the intial X-ray image (zeroth angle) and add it to the list of projections
    projection = np.array(gvxr.computeXRayImage());
    
    # Update the 3D visualisation
    gvxr.displayScene();

    # Retrieve the total energy
    energy_bins = gvxr.getEnergyBins(kwargs['Energy_units']) 
    photon_count_per_bin = gvxr.getPhotonCountEnergyBins();
    total_energy = 0.0;
    for energy, count in zip(energy_bins, photon_count_per_bin):
        total_energy += energy * count;  
    
    # Perform the flat-Field correction of raw data
    dark = np.zeros(projection.shape);

    flat = np.ones(projection.shape) * total_energy;
    projection = flat_field_normalize(projection,flat,dark)    
    write_image(kwargs['output_file'],projection,im_format=im_format,bitrate=8);


    theta.append(0.0);
    for i in range(1,num_projections):
        # Rotate the model by angular_step degrees
        for n,label in enumerate(mesh_names):
            gvxr.rotateNode(label, -1*angular_step, global_axis_vec[0], global_axis_vec[1], global_axis_vec[2]);
            total_rotation[2,n]+=angular_step
        # Compute an X-ray image and add it to the list of projections
        projection = np.array(gvxr.computeXRayImage());
        # Update the 3D visualisation
        gvxr.displayScene();
        theta.append(i * angular_step * math.pi / 180);
        projection = flat_field_normalize(projection,flat,dark)    
        write_image(kwargs['output_file'],projection,im_format=im_format,angle_index=i,bitrate=8);

    
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
        gvxr.renderLoop()
    #clear the scene graph ready for the next render in the loop    
    gvxr.removePolygonMeshesFromSceneGraph()
    return

def flat_field_normalize(arr, flat, dark, cutoff=None):
    """
    Normalize raw projection data using the flat and dark field projections.
    Again using numexpr to Speed up calculations over plain numpy.

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
    import numexpr as ne  
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
