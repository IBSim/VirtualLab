import itk
import numpy as np
import os

def Register_image(moving, static, **kwargs):
    # get kwargs or defaults if not set
    mask = kwargs.get("mask", None)
    Iterations = kwargs.get("Iterations", 5000)
    Samples = kwargs.get("Samples", 5000)
    resolutions = kwargs.get("Resolutions", 4)
    reg_params = kwargs.get("Reg_Params", None)
    im_series = kwargs.get("im_series", True)

    root, ext = os.path.splitext(moving)
    output_fname = f'{root}_R{ext}'
    if not im_series:
        # Single tiff stack
        fixed_image = itk.imread(static, itk.F)
        moving_image = itk.imread(moving, itk.F)
        if mask:
            fixed_mask = itk.imread(mask, itk.UC)
            fixed_mask = np.asarray(fixed_mask)
            fixed_mask = np.where(fixed_mask > 1, fixed_mask, 1)
            fixed_mask = itk.GetImageFromArray(fixed_mask)
        else:
            fixed_mask = None
    else:
        # image series
        moving_files = glob.glob(f"{moving}/*.tiff")
        static_files = glob.glob(f"{static}/*.tiff")
        ,fixed_image = np.zeros([1417,1417,len(static_files)])
        for static_im in static_files:
            with open(static_im) as f:
                fixed_image = itk.imread(f, itk.F)

        with open(moving_im) as f:
            moving_image = itk.imread(f, itk.F)
        if mask:
            fixed_mask = itk.imread(mask, itk.UC)
            fixed_mask = np.asarray(fixed_mask)
            fixed_mask = np.where(fixed_mask > 1, fixed_mask, 1)
            fixed_mask = itk.GetImageFromArray(fixed_mask)
        else:
            fixed_mask = None
    if reg_params != None:
        # load in list of a prams if given
        if not isinstance(reg_params, list):
            reg_params = [reg_params]
        parameter_object = itk.ParameterObject.New()
        parameter_object.ReadParameterFile(reg_params)
    else:
        # Import Default Parameter Map
        parameter_object = itk.ParameterObject.New()
        parameter_map_trans = parameter_object.GetDefaultParameterMap(
            "translation", resolutions
        )
        parameter_map_affine = parameter_object.GetDefaultParameterMap(
            "affine", resolutions
        )
        parameter_map_rigid = parameter_object.GetDefaultParameterMap(
            "rigid", resolutions
        )
        parameter_object.AddParameterMap(parameter_map_trans)
        parameter_object.AddParameterMap(parameter_map_rigid)
        parameter_object.AddParameterMap(parameter_map_affine)
        parameter_object.SetParameter("MaximumNumberOfIterations", str(Iterations))
        parameter_object.SetParameter("NumberOfSpatialSamples", str(Samples))

    # serialize each parameter map to a file.
    # for index in range(parameter_object.GetNumberOfParameterMaps()):
    #     parameter_map = parameter_object.GetParameterMap(index)
    #     parameter_object.WriteParameterFile(parameter_map,f"Parameters_{index}.txt")

    # Perform registration
    registered_image, params = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        fixed_mask=fixed_mask,
        parameter_object=parameter_object,
        log_to_console=True,
    )
    itk.imwrite(registered_image, output_fname)
