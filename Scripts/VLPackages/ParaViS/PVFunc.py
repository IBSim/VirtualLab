import os
import types

import numpy as np
import pvsimple

# ==================================================================================
# default settings colourbar and transfer function
colorbar_default = {'Orientation':'Vertical',
                    'Position' : [0.75, 0.05],
                    'ScalarBarLength' : 0.4,
                    'ScalarBarThickness' : 35,
                    'HorizontalTitle' : 1,
                    'TitleFontSize' : 20,
                    'TitleColor' : [0,0,0],
                    'TitleBold' : 1,
                    'RangeLabelFormat' : '%-#.1f',
                    'LabelFontSize' : 20,
                    'LabelColor' : [0,0,0],
                    'LabelBold' : 1,
                    'DrawTickLabels' : 0,
                    'DrawTickMarks' : 0
                    }

transferfunc_default = {'NumberOfTableValues' : 12}

screenshot_default = {'FontScaling':'Do not scale fonts',
                      'ImageResolution':[842, 542], 
                      'TransparentBackground':1}

# ==================================================================================
def GetEvalInfo():
    import SalomeFunc
    ArgDict = SalomeFunc.GetArgs()
    FncInfo = ArgDict['_PV_arg']
    return FncInfo

def FuncEval(FncInfo, add_funcs={}):

    available_funcs = {}
    available_funcs.update(add_funcs)
    if len(FncInfo)==0: return
    if type(FncInfo[0]) not in (list,tuple):
        FncInfo = [FncInfo] # make in to a list

    return_vals = []
    args,kwargs = [],{}
    for fnc_info in FncInfo:
        if len(fnc_info) == 1: # only a function name provided with no arguments
            funcname = fnc_info[0]
        elif len(fnc_info) == 2: # function name and argument
            funcname, args = fnc_info
        elif len(fnc_info) == 3: # function name, args and kwargs
            funcname, args, kwargs = fnc_info
        else:
            print('\n####################\n\nToo many values to unpack. Skipping {}\n\n####################\n'.format(fnc_info))
            continue

        if funcname not in available_funcs:
            print('\n####################\n\nFunction {} not found, so skipping\n\n####################\n'.format(funcname))
            continue

        func = available_funcs[funcname]
        ret = func(*args,**kwargs)
        return_vals.append(ret)

    return return_vals


# ==================================================================================
# useful functions
def OpenMED(ResFile):
    ''' Provide error message if medfile doesnt exist'''
    if not os.path.isfile(ResFile):
        print("File '{}' does not exist.".format(ResFile))
    medfile = pvsimple.MEDReader(FileName=ResFile)
    return medfile

def ObjectUpdate(obj,**kwargs):
    ''' Function for updating paraview objects (such as colourbar) using keyword arguments'''
    for key, val in kwargs.items():
        attr = getattr(obj,key)
        if type(attr) in (types.FunctionType, types.MethodType):
            # function which is evaluated using the arguments provided by the value
            attr(*val)
        else:
            # just an attribute of the object
            setattr(obj,key,val)
    return obj

def GetRenderView(camera=None):
    if camera is not None:
        # camera information is provided so use this
        render_view = Camera(*camera)
    else:
        # get default render view
        render_view = pvsimple.GetActiveViewOrCreate('RenderView')
    return render_view

# ==================================================================================
# results data
def GetDataRange(input,resname,restype):
    if restype.lower()=='point':
        data = input.PointData

    res_available = list(data.keys())
    if resname not in res_available:
        print("Warning: {} not in the list of available results")
        return

    ix = res_available.index(resname)
    range = data.GetArray(ix).GetRange()
    return range

def DataRange(input,resname):
    return GetDataRange(input,resname,'point')

# ==================================================================================
# Image capture

def Camera(FocalPoint,alpha1,alpha2,radius,ViewUp = [0,0,1]):
    ''' Positions camera using polar spherical coordinates.
        Focal points - focal point for camera
        alpha1 - rotation around z axis
        alpha2 - rotation from z axis 
        radius - distance of camera from focal point'''

    renderview = pvsimple.GetActiveViewOrCreate('RenderView')
    CameraPosition = [FocalPoint[0] + radius*np.cos(np.deg2rad(alpha2))*np.sin(np.deg2rad(alpha1)),
                      FocalPoint[1] - radius*np.cos(np.deg2rad(alpha2))*np.cos(np.deg2rad(alpha1)),
                      FocalPoint[2] + radius*np.sin(np.deg2rad(alpha2))]

    renderview.CameraPosition = CameraPosition
    renderview.CameraFocalPoint = FocalPoint
    renderview.CameraViewUp = ViewUp

    return renderview

def ImageCapture(renderView1, display, resname, filename, ResComponent='Res', CB={}, TF={}, Capture={}):
    # TODO: add in ability to colour by more than points
    pvsimple.ColorBy(display, ('POINTS', resname, ResComponent))

    display.SetScalarBarVisibility(renderView1, True)

    # Get transfer function for result 'resname'  
    surfLUT = pvsimple.GetColorTransferFunction(resname)
    # update default transfer function with parameters passed using TF
    _TF = {**transferfunc_default,**TF}
    surfLUT = ObjectUpdate(surfLUT,**_TF)

    # get color bar associated with the transfer function
    ColorBar = pvsimple.GetScalarBar(surfLUT, renderView1)
    # update default colourbar with parameters passed using CB
    _CB = {**colorbar_default,**CB} 
    ColorBar = ObjectUpdate(ColorBar,**_CB)

    # save
    _Capture = {**screenshot_default,**Capture}
    pvsimple.SaveScreenshot(filename, renderView1,**_Capture )
    ColorBar.Visibility = 0

    return locals()


# =========================================================================
# compare results

def _CompareSingleFile(source, resnames, image_paths,compare_ix=0,camera=None, render_view=None, res_type='Surface', CB={}, TF={}, Capture={}):
    '''
    Function used to compare results stores in the same source
    source - the data sources
    resnames - list, name of the results
    image_paths - list, paths to where the images will be saved
    '''
    # perform checks
    if len(resnames) != len(image_paths): # lengths are not compatible
        raise Exception("Length of result names and image file paths are not equal")

    pvsimple.HideAll()

    # if provided use the render view, else create one using the camera settings
    renderView1 = render_view if render_view is not None else GetRenderView(camera)

    display = pvsimple.Show(source, renderView1)
    display.Representation = res_type

    # get the range of values for the chosen result (default is the first) & add to transfer function dictionary
    range = DataRange(source, resnames[compare_ix])
    _TF = {'RescaleTransferFunction':range,**TF}
    
    # make images
    for resname,image_path in zip(resnames,image_paths):
        ImageCapture(renderView1, display, resname, image_path,
                    CB=CB,TF=_TF,Capture=Capture)

    pvsimple.Hide(source, renderView1) # hide display


def _CompareMultiFile(sources, resnames, image_paths,compare_ix=0, camera=None, render_view=None, res_type='Surface', CB={}, TF={}, Capture={}):
    '''
    Function used to compare results between multiple sources
    sources - the data sources
    resnames - list/string, name of the results
    image_paths - list, paths to where the images will be saved
    '''
    # perform checks
    if len(sources) != len(image_paths): # lengths of data sources and image paths are not compatible
        raise Exception("Length of data sources and image file paths are not equal")

    if type(resnames)==str:
        resnames = [resnames]*len(sources) # make in to list of appropriate length
    elif len(sources) != len(resnames): # lengths of resnames and sources must be the same length
        raise Exception("Lengths of data sources and result names must be equal")

    pvsimple.HideAll()

    # if provided use the render view, else create one using the camera settings
    renderView1 = render_view if render_view is not None else GetRenderView(camera)

    # get the range of values for the chosen result (default is the first) & add to transfer function dictionary.
    range = DataRange(sources[compare_ix],resnames[compare_ix])
    _TF = {'RescaleTransferFunction':range,**TF}

    # capture images
    for source,resname,image_path in zip(sources,resnames,image_paths):
        display = pvsimple.Show(source, renderView1)
        display.Representation = res_type
        ImageCapture(renderView1,display,resname,image_path,
                    CB=CB,TF=_TF,Capture=Capture)
        pvsimple.Hide(source, renderView1)

def Compare(source, resnames, image_paths, **kwargs):
    if type(source) in (tuple,list):
        # results are in a multiple files
        _CompareMultiFile(source, resnames, image_paths,**kwargs)
    else :
        # a single results file with many results
        _CompareSingleFile(source, resnames, image_paths,**kwargs)            

# =========================================================================
# difference results

def _DiffCapture(source, resnames, image_path, camera=None, render_view=None, relative=False, absolute_difference=False, 
                 res_type='Surface', diff_resname='Diff', zero_centre=False, CB={}, TF={}, Capture={}):

    ''' 
    Function to capture image of the difference between two results.
    source - paraview data source
    resnames - name of the results to calculate the difference between
    image_path - path to where the image will be saved
    relative - whether or not the difference is scaled by the result
    '''

    pvsimple.HideAll()

    # if provided use the render view, else create one using the camera settings
    renderView1 = render_view if render_view is not None else GetRenderView(camera)

    # calculation which will be made 
    if relative: #scale by the second result
        calc_string = '({0}-{1})/{1}'.format(*resnames) # (result 1 - result 2 scaled by result 2)
    else: # dont scale by results
        calc_string = '{}-{}'.format(*resnames) # (result 1 - result 2)

    if absolute_difference: # give the absolute difference
        calc_string = 'abs({})'.format(calc_string)

    # use calculator filter to make result
    calculator1 = pvsimple.Calculator(Input=source)
    calculator1.Function = calc_string
    calculator1.ResultArrayName = diff_resname

    # show result
    display = pvsimple.Show(calculator1, renderView1)
    display.Representation = res_type

    # update transfer function depending on whether or not the data is centred
    if zero_centre:
        Range = DataRange(calculator1,diff_resname)
        maxabs = np.abs(Range).max()
        Range = [-maxabs,maxabs ]
        _TF = {'RescaleTransferFunction':Range,**TF}
    else:
        _TF = {**TF}

    # capture image
    ImageCapture(renderView1, display, diff_resname, image_path,
                 CB=CB,TF=_TF,Capture=Capture)

    # hide result
    pvsimple.Hide(calculator1, renderView1)

def _DiffSingle(source, resnames, image_path, **kwargs):
    '''
    function used when both results are in a single source
    '''
    # =============================================================
    # perform checks    
    if len(resnames) != 2: # incorrect number of result names given
        raise Exception("Number of results must be equal to 2")
    
    # =============================================================
    _DiffCapture(source,resnames,image_path, **kwargs)

def _DiffMulti(sources, resnames, image_path, **kwargs):
    '''
    function used when results are in two different sources
    '''
    # =============================================================
    # perform checks
    if len(sources) != 2: # lengths of result names and filenames are not compatible
        raise Exception("Number of sources must be equal to 2")
    if type(resnames) == tuple: 
        resnames = list(resnames) # as we will be indexing
    if type(resnames) == list and len(resnames) !=2:
        # lengths of resnames and results must be the same length
        raise Exception("Number of results must be equal to 2")
    elif type(resnames) == str:
        pass
    else:
        raise TypeError('resnames must be a string or a list')

    # =============================================================
    # combine results from multiple sources in to one
    combined_source = pvsimple.AppendAttributes(Input=sources)
    # format names
    if type(resnames)==str:
        resnames = [resnames,'{}_input_1'.format(resnames)]
    elif resnames[0]==resnames[1]:
        resnames = [resnames,'{}_input_1'.format(resnames)]

    # =============================================================
    _DiffCapture(combined_source, resnames, image_path, **kwargs)

def Difference(source, resnames, filenames, **kwargs):
    '''
    Function to capture image of the difference between two results.
    source - paraview data source(s)
    resnames - name(s) of the results to calculate the difference between
    image_path - path to where the image will be saved
    kwargs - see _DiffCapture
    '''

    if type(source) in (tuple,list): # results are in a multiple files
        _DiffMulti(source, resnames, filenames,**kwargs)
    else : # a single results file containing both results
        _DiffSingle(source, resnames, filenames,**kwargs)     

def RelativeDifference(*args, **kwargs):
    kwargs['relative'] = True
    return Difference(*args,**kwargs)

def CompareAndDiff(source, resnames, image_paths_compare, image_path_diff,compare_kwargs={},diff_kwargs={}):
    # do difference first as the error catching is better
    Difference(source,resnames,image_path_diff,**diff_kwargs)
    Compare(source,resnames,image_paths_compare,**compare_kwargs)


