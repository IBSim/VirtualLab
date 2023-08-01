
import numpy as np
from Scripts.Common.tools import MEDtools

def GetVolume(VolumeCoords):
    # returns the volume of each volume element
    v1 = VolumeCoords[:,1] - VolumeCoords[:,0]
    v2 = VolumeCoords[:,2] - VolumeCoords[:,0]
    v3 = VolumeCoords[:,3] - VolumeCoords[:,0]
    cross = np.cross(v1,v2)
    dot = (cross*v3).sum(axis=1)
    Volumes = 1/float(6)*np.abs(dot) # volume of each volume element
    return Volumes

def PowerByVolume(VolumeCoords,WattsPV):
    # returns the amount of power delivered to each volume element
    Volumes = GetVolume(VolumeCoords)
    Power = Volumes*WattsPV 
    return Power

def TotalPower(VolumeCoords,WattsPV):
    # returns the total power delivered to a component
    Power = PowerByVolume(VolumeCoords,WattsPV)
    return Power.sum()

# TODO - add these two to MeshInfo class
def NodeToElem(Connectivity,NodalValues):
    'Get values for the nodes which make up the element and average it out'
    return NodalValues[Connectivity].mean(axis=1)

def ElementCoordinates(Connectivity,Coordinates):
    'If connectivity is 100 x 4 & coordinates are 3D then this function returns a matrix of shape 100 x 4 x 3'
    return Coordinates[Connectivity] # get coordinate for each node of element in the shape of connect

def ElementCoordinates2(Connectivity,NodeCoords,NodeNum=None):
    ''' 
    Input:
    Connectivity: shape (m x p), where m is the number of elements and p is the number of points which make up the element.
    NodeCoords: shape (n x d), where n is the number of nodes which make up the mesh connectivity and d is the dimension.
    NodeNum: shape (n,), optional list to relate the corrdinate to a global reference system
    
    returns:
    tensor of shape (m x p x d) which stores coordinate information in the shape of the connectivity
    '''
    n = len(NodeCoords)
    if NodeNum is not None:
        if type(NodeNum) == list: 
            NodeNum = np.array(NodeNum)
        assert len(NodeNum)==n

    if NodeNum is None or (NodeNum==np.arange(len(NodeCoords))).all(): 
        # assume NodeNum runs from 0 to n-1
        ElemCoord = NodeCoords[Connectivity] 
    else:
        # convert connectivity to local numbering system
        local_ix = np.searchsorted(NodeCoords,Connectivity)
        ElemCoord = NodeCoords[local_ix]        
    
    return ElemCoord


def TotalPowerMED(ERMESResFile,GroupName=None):
    meshdata = MEDtools.MeshInfo(ERMESResFile)

    Coords = meshdata.GetNodeXYZ('all') 
    Connect = meshdata.Connectivity(GroupName)
    Connect = Connect - 1 # subtract 1 as python index starts from 0
    # coordinates in element shape
    elem_cd = ElementCoordinates(Connect,Coords) # get coordinate for each node of element in the shape of connect
    # get nodal values and convert to element values
    JH_Node = MEDtools.NodalResult(ERMESResFile,'Joule_heating')
    JH_vol = NodeToElem(Connect,JH_Node)

    Power = TotalPower(elem_cd,JH_vol)

    return Power


def MeshGlobal2Local(Connectivity,NodeNum):
    ''' Converts a global mesh connectivity to a local one based on the index of NodeNum'''
    # checks 
    return np.searchsorted(NodeNum,Connectivity)

def _VariationOrig(ElemRes,elem_cd,surface_norm):
    m,p = ElemRes.shape

    # get a vector of values augmented by the joule heating value, which is added in the direction of the urface normal
    augmented = []
    for i in range(p):
        a = surface_norm*ElemRes[:,i:i+1]
        augmented.append(a)
    augmented = np.swapaxes(np.array(augmented),0,1)

    # calculate the cross between the surface normal and the new augemnted normal
    elem_cd_aug = elem_cd + augmented
    v1,v2 = elem_cd_aug[:,1] - elem_cd_aug[:,0], elem_cd_aug[:,2] - elem_cd_aug[:,0]
    cross = np.cross(v1,v2)
    _cross = np.cross(cross,surface_norm)
    var = np.linalg.norm(_cross,axis=1).sum()

    return var


def VariationOrig(NodeRes,Connectivity,NodeCoords,NodeNum=None):

    m,p = Connectivity.shape
    n = len(NodeRes)
    n2,d = NodeCoords.shape

    assert n==n2
        
    if NodeNum is not None:
        Connectivity = MeshGlobal2Local(Connectivity,NodeNum)

    elem_cd = NodeCoords[Connectivity]
    # get unit normal to surface of each element

    v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
    surface_norm = np.cross(v1,v2)
    surface_norm = surface_norm/np.linalg.norm(surface_norm,axis=1)[:,None] # unit normal

    if NodeRes.ndim==1:
        ElemRes = NodeRes[Connectivity] # results in the shape of the connectivity
        var = _VariationOrig(ElemRes,elem_cd,surface_norm)
    else:
        var = []
        for _NodeRes in NodeRes:
            ElemRes = _NodeRes[Connectivity]
            _var = _VariationOrig(ElemRes,elem_cd,surface_norm)
            var.append(_var)
        var = np.array(var)

    return var

def _Variation(a1,a2,v1,v2):
    g = a2[:,None]*v1 - a1[:,None]*v2
    var = np.linalg.norm(g,axis=1) # work out the variation for each element
    var = var.sum() # sum up all variations
    return var

def Variation(NodeRes,Connectivity,NodeCoords,NodeNum=None):
    ''' A simplified implementation of VariationOrig. This returns the same value but for much less computation'''
    m,p = Connectivity.shape # m is the number of elements, p is the number of points hwich make up the element
    n,d = NodeCoords.shape # n is the number of nodes which make up the mesh, d is the dimension
    n2 = NodeRes.shape[-1]

    if n!=n2:
        raise Exception('The number of nodes in the results ({}) are not equal to the number of nodes which make up the mesh ({})'.format(n2,n))
        
    if NodeNum is not None:
        Connectivity = MeshGlobal2Local(Connectivity,NodeNum)

    elem_cd = NodeCoords[Connectivity]
    elem_res = (NodeRes.T[Connectivity]).T # compatibility with multiple variation values

    v1,v2 = elem_cd[:,1] - elem_cd[:,0], elem_cd[:,2] - elem_cd[:,0]
    a1,a2 = elem_res[:,1] - elem_res[:,0], elem_res[:,2] - elem_res[:,0]

    if NodeRes.ndim==1:
        var = _Variation(a1,a2,v1,v2)
    else:
        var = []
        for i in range(NodeRes.shape[0]):
            _var = _Variation(a1[i],a2[i],v1,v2)
            var.append(_var)
        var = np.array(var)
    return var

def VariationMED(ERMESResFile,SurfaceName):

    meshdata = MEDtools.MeshInfo(ERMESResFile)
    surface_data = meshdata.GroupInfo(SurfaceName)
    surface_coords = meshdata.GetNodeXYZ(surface_data.Nodes)
    meshdata.Close()

    JH_Node = MEDtools.NodalResult(ERMESResFile,'Joule_heating',GroupName=SurfaceName)
    
    var = Variation(JH_Node,surface_data.Connect,surface_coords,surface_data.Nodes)
    return var



