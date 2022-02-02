import os
import numpy as np
from scipy import spatial, special
from Scripts.Common.VLFunctions import MeshInfo

def Uniformity2(JHNode, MeshFile):
    Meshcls = MeshInfo(MeshFile)
    CoilFace = Meshcls.GroupInfo('CoilFace')
    Area, JHArea = 0, 0 # Actual area and area of triangles with JH
    for nodes in CoilFace.Connect:
        vertices = Meshcls.GetNodeXYZ(nodes)
        # Heron's formula
        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        Area += area

        vertices[:,2] += JHNode[nodes - 1].flatten()

        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area1 = np.sqrt(s * (s - a) * (s - b) * (s - c))
        JHArea += area1
    Meshcls.Close()

    Uniformity = JHArea/Area
    return Uniformity

def Uniformity3(JHNode, MeshFile):
    Meshcls = MeshInfo(MeshFile)
    CoilFace = Meshcls.GroupInfo('CoilFace')

    Area, JHVal = 0, [] # Actual area and area of triangles with JH
    for nodes in CoilFace.Connect:
        vertices = Meshcls.GetNodeXYZ(nodes)
        # Heron's formula
        a, b, c = spatial.distance.pdist(vertices, metric="euclidean")
        s = 0.5 * (a + b + c)
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        Area += area

        v = np.mean(JHNode[nodes])

        JHVal.append(area*v)

        vertices[:,2] += JHNode[nodes - 1].flatten()

    Meshcls.Close()

    Variation = np.std(JHVal)
    return Variation
