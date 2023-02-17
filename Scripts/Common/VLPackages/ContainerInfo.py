import os
import sys
from importlib import reload, import_module
from types import SimpleNamespace as SN

this_dir = os.path.dirname(os.path.abspath(__file__))

def _GetInfo(PackageName):

    path = "{}/{}".format(this_dir,PackageName)

    config_fname = "ContainerConfig"
#    if not os.path.isfile("{}/{}.py".format(path,config_fname)):
#        # package has no container config argument
#        print('Error')
#        continue
        
    sys.path.insert(0,path)
    contconfig = reload(import_module(config_fname))
    sys.path.pop(0)
    
    containers_avail = contconfig.__all__
  
    cont_dict = {}    
    for container_name in containers_avail:
        # check container_name is in contconfig
        info = getattr(contconfig,container_name)
        cont_dict[container_name] = SN(**info)
        
    return cont_dict
    
def GetInfo(PackageName):
    cont_dict = _GetInfo(PackageName)
    return list(cont_dict.values())[0]

            
        


class Containers():
    def __init__(self):
        self.get_cont_info()
        
    def get_cont_info(self,):
    
        self.defaults = {}
        self.containers = {}
        
        for content in os.listdir(this_dir):
            path = "{}/{}".format(this_dir,content)
            if not os.path.isdir(path):
                # skip files
                continue

            config_fname = "ContainerConfig"
            if not os.path.isfile("{}/{}.py".format(path,config_fname)):
                # package has no container config argument
                continue
                
            sys.path.insert(0,path)
            contconfig = reload(import_module(config_fname))
            sys.path.pop(0)
            
            containers_avail = contconfig.__all__
          
            cont_dict = {}    
            for container_name in containers_avail:
                # check container_name is in contconfig
                info = getattr(contconfig,container_name)
                cont_dict[container_name] = SN(**info)
 
            self.containers[content] = cont_dict
            self.defaults[content] = containers_avail[0] # default is the first in the list

    def _check_package(self, PackageName):
        # Check PackageName is available
        if PackageName not in self.containers:
            print('Package {} not available.\nPlease check if the appropriate config file is created'.format(PackageName))
            return 1

        
    def _check_container(self,PackageName,ContainerName):
    
        err = self._check_package(PackageName,ContainerName)
        if err: return err
        
        # Check container name is available            
        containers = self.containers[PackageName]
        if ContainerName not in containers:
            print('Container {} not available for package {}'.format(ContainerName,PackageName))
            return 1

        
    def GetDefault(self,PackageName):
        return self.defaults[PackageName]
                 
    def SetContainer(self,PackageName,ContainerName):
        if self._check_container(PackageName,ContainerName):
            print('Container has not been set')
            return
    
        self.defaults[PackageName] = ContainerName
        
    def GetContainer(self, PackageName, ContainerName=None):
    
        if ContainerName is None:
            ContainerName = self.GetDefault(PackageName)
            
        return self.containers[PackageName][ContainerName]
       
#a = Containers()
#print(a.GetDefault('Salome'))
#print(a.GetContainer('Salome'))

