
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

def Compare(med_result, resnames, filenames, **kwargs):
    if type(med_result)==str:
        # results are in a single file
        CompareSingleFile(med_result, resnames, filenames,**kwargs)
    elif type(med_result) in (tuple,list):
        # results are in a multiple files
        CompareMultiFile(med_result, resnames, filenames,**kwargs)
            
