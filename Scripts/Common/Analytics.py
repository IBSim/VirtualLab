import requests
from datetime import datetime

'''
This function serves to provide a small amount of analytics data on how
VirtualLab is used. VirtualLab is open-source software which has been developed
through the support of research grants. The small amount of statistics this
function gathers enables us to evidence the impact and value of our research.
This is invaluable when we apply for funding which will lead to further
development of VirtualLab for your benefit.

The anonymised data which is sent to us is the following: frequency of use;
which virtual experiment module was used; whether this was run in single or
parametric mode; the number of computing cores used.

We hope you agree to this data being sent to us to assist us in applying for
future research grants. If you would like to disable this feature, this may be
done in VLconfig.py.
'''

def event(envdict):
    tracking_id = 'UA-112907949-3' #tid
    clientid_str = str(datetime.now()) #cid
    campaign_name_str = 'UbuntuVM' #cn
    key1 = 'UbuntuVM' #key1
    event_category = envdict['Simulation']
    event_action = "{}_{}_{}".format(envdict['NumSims'],envdict['NumMeshes'],envdict['NumMeshesCr'])
    event_label = "{}_{}".format(envdict['ncpus'],envdict['mpi_nbcpu'])
    tracking_url = 'https://www.google-analytics.com/collect?v=1&t=event&tid='+tracking_id+'&cid='+clientid_str+'&ec='+event_category+'&ea='+event_action+'&el='+event_label+'&key1='+key1+'&aip=0'
    try:
        requests.post(tracking_url)
    except :
        pass
    # Useful urls
    # https://www.themarketingtechnologist.co/measure-your-python-projects-with-google-analytics/
    # https://www.optimizesmart.com/understanding-universal-analytics-measurement-protocol/
    # https://requests.readthedocs.io/en/master/user/quickstart/
    # https://developers.google.com/analytics/devguides/collection/protocol/v1/reference
