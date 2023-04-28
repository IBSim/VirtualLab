import itk
import numpy as np
def Register_image(moving,static,mask,output_im,Iterations=2500,resolutions=4):
    fixed_image = itk.imread(static,itk.F)
    moving_image = itk.imread(moving,itk.F)
    fixed_mask = itk.imread(mask,itk.UC)
    fixed_mask = np.asarray(fixed_mask)
    fixed_mask = np.where(fixed_mask>1,fixed_mask,1)
    fixed_mask = itk.GetImageFromArray(fixed_mask)
    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()
    # parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid',resolutions)
    # parameter_map_rigid['MaximumNumberOfIterations'] = [str(Iterations)]
    # Match to conservative registration values
    parameter_map_trans = parameter_object.GetDefaultParameterMap('translation',resolutions)
    parameter_map_trans['MaximumNumberOfIterations'] = [str(Iterations)]
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine',resolutions)
    parameter_map_affine = parameter_object.GetDefaultParameterMap('rigid',resolutions)
    parameter_map_affine['MaximumNumberOfIterations'] = [str(Iterations)]
    parameter_map_affine['BSplineInterpolatonOrder'] = ['3']
    parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline',resolutions,20.0)
    parameter_map_bspline['MaximumNumberOfIterations'] = [str(Iterations)]
    parameter_map_bspline['BSplineInterpolatonOrder'] = ['3']
    parameter_object.AddParameterMap(parameter_map_trans)
    parameter_object.AddParameterMap(parameter_map_affine)
    parameter_object.AddParameterMap(parameter_map_bspline)
    parameter_object.SetParameter('NumberOfHistogramBins', '128')
    parameter_object.SetParameter('NumberOfSpatialSamples','5000')
# serialize each parameter map to a file.
    # for index in range(parameter_object.GetNumberOfParameterMaps()):
    #     parameter_map = parameter_object.GetParameterMap(index)
    #     parameter_object.WriteParameterFile(parameter_map,f"Parameters_{index}.txt")
    registered_image, params = itk.elastix_registration_method(fixed_image, moving_image,fixed_mask=fixed_mask,parameter_object=parameter_object,log_to_console=True)
    itk.imwrite(registered_image,output_im)
    # reg = np.asarray(registered_image).astype(np.uint32)
    # fixed = np.asarray(fixed_image).astype(np.uint32)

if __name__== '__main__':
    work_dir = '/media/ben/0ab14e4f-9a0f-460b-965c-48c0e28b833f/Reg_testing/'
    #moving = work_dir + 'HIVE_Wonky-down_50.tiff'
    #static = work_dir + 'HIVE_GVXR-down_50.tiff'
    moving = work_dir + 'Survos_Down_wonkey.tif'
    static = work_dir + 'Survos_Down.tif'
    output_file = 'result_image.tiff'
    mask = work_dir + 'HIVE_cad_Down.tif'
    Register_image(moving,static,mask,output_file,Iterations=5000,resolutions=6)
