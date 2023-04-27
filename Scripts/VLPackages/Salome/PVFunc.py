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

def ImageCapture(renderView1, meddisplay, resname, filename, ResComponent='Res', CB={}, TF={}, Capture={}):
    # TODO: add in ability to colour by more than points
    pvsimple.ColorBy(meddisplay, ('POINTS', resname, ResComponent))

    meddisplay.SetScalarBarVisibility(renderView1, True)

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

def CompareSingleFile(medfile, resnames, filenames,compare_ix=0,camera=None, render_view=None, res_type='Surface', CB={}, TF={}, Capture={}):
    # perform checks
    if len(resnames) != len(filenames):
        # lengths of result names and filenames are not compatible
        raise Exception("Lengths of result names and file names must be equal")

    pvsimple.HideAll()

    # get render view
    if render_view is not None:
        renderView1 = render_view # use the render view provided
    else:
        renderView1 = GetRenderView(camera) # get renderview

    meddisplay = pvsimple.Show(medfile, renderView1)
    meddisplay.Representation = res_type

    # get the range of values for the chosen result (default is the first) & add to transfer function dictionary
    range = DataRange(medfile, resnames[compare_ix])
    _TF = {'RescaleTransferFunction':range,**TF}
    
    # make images
    for res,fname in zip(resnames,filenames):
        ImageCapture(renderView1,meddisplay,res,fname,CB=CB,TF=_TF,Capture=Capture)

    meddisplay = pvsimple.Hide(medfile, renderView1) # hide display

def CompareMultiFile(medfiles, resnames, filenames,compare_ix=0, camera=None, render_view=None, res_type='Surface', CB={}, TF={}, Capture={}):
    if len(medfiles) != len(filenames):
        # lengths of med results and filenames are not compatible
        raise Exception("Lengths of results and file names must be equal")

    if type(resnames)==str:
        resnames = [resnames]*len(medfiles) # make in to list of appropriate length
    elif len(medfiles) != len(resnames):
        # lengths of resnames and results must be the same length
        raise Exception("Lengths of results and result names must be equal")


    pvsimple.HideAll()

    # get render view
    if render_view is not None:
        renderView1 = render_view # use the render view provided
    else:
        renderView1 = GetRenderView(camera) # get renderview

    # get the range of values for the chosen result (default is the first) & add to transfer function dictionary.
    range = DataRange(medfiles[compare_ix],resnames[compare_ix])
    _TF = {'RescaleTransferFunction':range,**TF}

    # capture images
    for medfile,resname,fname in zip(medfiles,resnames,filenames):
        meddisplay = pvsimple.Show(medfile, renderView1)
        meddisplay.Representation = res_type
        ImageCapture(renderView1,meddisplay,resname,fname,CB=CB,TF=_TF,Capture=Capture)
        pvsimple.Hide(medfile, renderView1)

def Compare(results, resnames, filenames, **kwargs):
    if type(results) in (tuple,list):
        # results are in a multiple files
        CompareMultiFile(results, resnames, filenames,**kwargs)
    else :
        # a single results file with many results
        CompareSingleFile(results, resnames, filenames,**kwargs)            

def DifferenceSingleFile(medfile, resnames, filename, camera=None, render_view=None, absolute_difference=False, 
                         res_type='Surface', diff_resname='Difference', zero_centre=False, CB={}, TF={}, Capture={}):
    if len(resnames) != 2:
        # incorrect number of result names given
        raise Exception("Number of results must be equal to 2")

    pvsimple.HideAll()

    # get render view
    if render_view is not None:
        renderView1 = render_view # use the render view provided
    else:
        renderView1 = GetRenderView(camera) # get renderview

    # calculate difference
    calculator1 = pvsimple.Calculator(Input=medfile)
    func_str = '{}-{}'.format(*resnames)
    if absolute_difference:
        func_str = 'abs({})'.format(func_str)
    calculator1.Function = func_str
    calculator1.ResultArrayName = diff_resname

    meddisplay = pvsimple.Show(calculator1, renderView1)
    meddisplay.Representation = res_type

    # enable data to be centred around zero
    if zero_centre:
        Range = DataRange(calculator1,diff_resname)
        maxabs = np.abs(DataRange).max()
        DataRange = [-maxabs,maxabs ]
        _TF = {'RescaleTransferFunction':DataRange,**TF}
    else:
        _TF = {**TF}

    # capture image
    ImageCapture(renderView1,meddisplay,diff_resname,filename,CB=CB,TF=_TF,Capture=Capture)

    # hide result
    meddisplay = pvsimple.Hide(calculator1, renderView1) 

def DifferenceMultiFile(medfiles, resnames, filename, camera=None, render_view=None, absolute_difference=False, 
                         res_type='Surface', diff_resname='Difference', CB={}, TF={}, Capture={}):
    # perform checks
    if len(medfiles) != 2:
        # lengths of result names and filenames are not compatible
        raise Exception("Number of results must be equal to 2")

    if type(resnames)==str:
        resnames = [resnames,'{}_input_1'.format(resnames)]
    elif type(resnames) in (list,tuple) and len(resnames) !=2:
        # lengths of resnames and results must be the same length
        raise Exception("Lengths of results and result names must be equal")
    
    pvsimple.HideAll()

    # get render view
    if render_view is not None:
        renderView1 = render_view # use the render view provided
    else:
        renderView1 = GetRenderView(camera) # get renderview

    # calculate difference
    appendAttributes1 = pvsimple.AppendAttributes(Input=medfiles)
    calculator1 = pvsimple.Calculator(Input=appendAttributes1)

    func_str = '{}-{}'.format(*resnames)
    if absolute_difference:
        func_str = 'abs({})'.format(func_str)
    calculator1.Function = func_str
    calculator1.ResultArrayName = diff_resname

    # enable data to be centred around zero
    DataRange = DataRange(calculator1,diff_resname)
    if zero_centre:
        maxabs = np.abs(DataRange).max()
        DataRange = [-maxabs,maxabs ]
    _TF = {'RescaleTransferFunction':DataRange,**TF}


    meddisplay = pvsimple.Show(calculator1, renderView1)
    meddisplay.Representation = res_type

    # capture image
    ImageCapture(renderView1,meddisplay,diff_resname,filename,CB=CB,TF=_TF,Capture=Capture)

    # hide result
    meddisplay = pvsimple.Hide(calculator1, renderView1)     

def Difference(results, resnames, filenames, **kwargs):
    if type(results) in (tuple,list):
        # results are in a multiple files
        DifferenceMultiFile(results, resnames, filenames,**kwargs)
    else :
        # a single results file with many results
        DifferenceSingleFile(results, resnames, filenames,**kwargs)     