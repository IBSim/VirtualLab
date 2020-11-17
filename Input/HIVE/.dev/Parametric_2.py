from types import SimpleNamespace as Namespace

Mesh = Namespace()
Sim = Namespace()
Mesh.Name = ['TestOut1','TestOut2']
Mesh.CoilDisplacement = [[0,0,0.003], [0,0,0.002]]

Sim.Name = ['TestOut1','TestOut2']
Sim.Mesh = ['TestOut1','TestOut1']
Sim.Current = [2000,2000]
Sim.Run = [1,0]
