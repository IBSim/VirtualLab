from types import SimpleNamespace as Namespace

modelib = Namespace()

modelib.Name = 'microstructure'

modelib.File='DDD'


modelib.dislocationline = 2e14
modelib.dislocationloop = 1e22
modelib.prec=1e21
modelib.b=.1
modelib.dim=1
modelib.temp=300
modelib.strainrate=1e-11

DPA = Namespace()

DPA.Name = 'microstructure'

DPA.File=('mechanical_load_results','dpa_calculation')

