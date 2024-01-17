import pandas as pd
import os
import json
import numpy as np
import h5py
import scipy.stats
import matplotlib.pyplot as plt
from Scripts.VLPackages.Salome.API import Run as SalomeRun
from Scripts.VLPackages.ParaViS import API as ParaVis
import sys
sys.dont_write_bytecode=True

import numpy as np
from importlib import import_module



def Single(VL,DADict):
    Parameters = DADict['Parameters']
    
    ResDir = "{}/{}/Aster/".format(VL.PROJECT_DIR, DADict["_Name"])
    case = "test"
    ext = ".csv"
    case1='yield'
    ext = ".csv"
  
    conditions = ["TMax_tn", "VMMax_tn", "TMin_tn"]
 
   
    
    for condition in conditions:
        f= open(ResDir +'plasticflow'+ condition+'.txt','w')
        f.close()
        fname_case = ResDir + case + '_' + condition + ext
        fname_case1 = ResDir + case1 + '_' + condition + ext
     
        tf1 = pd.read_csv(fname_case)
        
        tf1 = tf1.dropna()
        yf1 = pd.read_csv(fname_case1)
        
        yf1 = yf1.dropna()
       
        column_yield=np.array(yf1['rth41___FLUX_NOEU'].values)
        
        column_temp=np.array(tf1['rth_____TEMP'].values)
        
        cv=np.stack([column_temp,column_yield,column_yield],axis=1)
        
        with open(ResDir +'plasticflow'+ condition+'.txt','a') as f:
        
            np.savetxt(f, cv, delimiter="\t")
     
        tf1=pd.read_csv(ResDir +'plasticflow'+ condition+'.txt',header= None)
    
    for condition in conditions:    
        np.savetxt(ResDir +'plasticflow'+ condition+'.txt', tf1.values, delimiter=" ",fmt='%s',comments='',header="temp\tSe(un)\tSe(ir)")
      
        
 
    
    for condition in conditions:
        
        ductility_values = plasticflow = ResDir +'plasticflow'+ condition+'.txt'
        fname_out = ResDir + case + '_' + condition + ".json"
    
        subsection = 'Plastic_flow_localisation'
    
        write_json(subsection, fname_out)  
        fname_case = ResDir + case + '_' + condition + ext
        
        # Need to read text file into array 
        df1 = create_array_csv(fname_case)
        df2 = create_array_txt(ductility_values)
        
        # Creating the unirridated and irrdated Rf values  
        # for this specific case 
        values = Rf_values(df1, df2, condition)
    
        # Appends the data to an existing JSON file under 
        # Plastic flow localisation 
       
        append_json(values, fname_out) 
      
       
    for condition in conditions: 
       
        values_irr = []
        fname_out = ResDir + case + '_' + condition + ".json"
        df = pd.read_json(fname_out, orient = 'columns')
        df = df["Plastic_flow_localisation"]
        i=0
      
        item = df[i][condition]
        print(item)
        value_irr = item[0]['Strength_usage']
 
        values_irr.append(100/value_irr)
        
   
        data = {getattr(Parameters, 'days'): values_irr}
       
        labels = [condition]
        if condition =="TMax_tn":
           labels = ["Maximum Temperature"]
           df3 = pd.DataFrame(data,columns=[getattr(Parameters, 'days')], index = labels)
        if condition=="VMMax_tn":
           labels = ["Maximum Stress"]
           df1 = pd.DataFrame(data,columns=[getattr(Parameters, 'days')], index = labels)
        if condition=="TMin_tn":
           labels = ["Minimum Temperature"]
           df2 = pd.DataFrame(data,columns=[getattr(Parameters, 'days')], index = labels)

    s=pd.concat([df3,df1,df2])
    ax=s.plot.barh(width = 0.4,figsize = (24,12), edgecolor = 'black')

    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    bars = ax.patches
    hatches = ''.join(h*len(s) for h in '|/x.')

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams.update({'font.family':"Times New Roman"})

    hfont = {'fontname':'Times New Roman'}

    plt.xlabel('Strength Usage (%)', fontsize = 15,**hfont,labelpad=5)

    #plt.legend(frameon=False,  fontsize = 30)
    plt.tick_params(axis="y", labelsize="15")
    plt.tick_params(axis="x", labelsize="15")
    plt.tick_params(top = 'on', right = 'on', direction = 'in', length = 8)
    plt.gca().axes.get_xaxis().set_ticks([0,20,40,60,80,100,120,140,160,180,200])


    plt.savefig(ResDir + '/' + 'lifecycle.png', dpi = 500, pad_inches = 0)
    plt.show(block=True);

# function that writes to an existing JSON fname
def write_json(subsection, fname_out):
    subsection ={
            '{}'.format(subsection):[]
            }
    with open(fname_out,'w') as fname:
         json.dump(subsection, fname, indent = 4)
         
         
# function that creates an array of the data from the text/csv files
def create_array_csv(fname):
    fname = pd.read_csv(fname)
    fname = fname.apply(pd.to_numeric, errors='coerce')
    fname = fname.dropna()
    return fname


def create_array_txt(fname):
    fname = pd.read_csv(fname, sep = "\t")
    fname = fname.apply(pd.to_numeric, errors='coerce')
    fname = fname.dropna()
    return fname

def Rf_values(dataframe, ductility_values, case):
   
    # in the following lines we create arrays whose entries are the 
    # corresponding entries of the columns named "variable"...
    temperature = np.array(dataframe['rth_____TEMP'].values)
    # Stress Magnitude is equivalent to VonMises stress here 
    VonMises = np.array(dataframe['P1______SIEQ_NOEU'].values)
    arc_length = np.array(dataframe['arc_length'].values)
    
    # linear interpolation using scipy
    m, c, r, p, se = scipy.stats.linregress(VonMises, arc_length)

    # choosing the first and last elements of the arc length
    first = arc_length[0]
    last = arc_length[-1]
  
    # caluclating the maximum and minimum pressures along the x-axis
    x_pmin = (first - c)/m
    x_pmax = (last - c)/m

    # takes the average pressure along the x-axis, changing to MPa 
    avg_x = (x_pmin + x_pmax)/2
    pressure = avg_x
    
    #  average temperature 
    Tm = np.average(temperature)

    # creates arrays containing data from txt file about properties of alloy
    # here un and ir represent unirridated and irridated respectively 
    x =  np.array(ductility_values['temp'].values)
    y_un = np.array(ductility_values['Se(un)'].values)
    y_ir =  np.array(ductility_values['Se(ir)'].values)
    
    # linear interpolation at Tm to find values for Se 
    Se_un = np.interp(Tm, x, y_un)
    Se_ir = np.interp(Tm, x, y_ir)
    
    # ratio factor Rf (we want this ratio to be greater than one)
    Rf_un = Se_un/pressure 
    Rf_ir = Se_ir/pressure 
  
    # name of the object that will be appended to the JSON file 
 
    
    # object that will be appended to the JSON file
    values ={
            "{}".format(case):
                [{'Strength_usage' : Rf_un},
                {'Strength_usage': Rf_ir}]
            }
            
    return values
    
def append_json(new_data, fname_out):
    with open(fname_out,'r+') as fname:
        # First we load existing data into a dict.
        fname_data = json.load(fname)
        # Join new_data with fname_data inside 
        fname_data["Plastic_flow_localisation"].append(new_data)
        fname.seek(0)
        json.dump(fname_data, fname, indent = 4)       
