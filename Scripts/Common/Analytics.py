import requests
from datetime import datetime

def event(envdict):
    tracking_id = 'UA-112907949-3' #tid
    clientid_str = str(datetime.now()) #cid
    campaign_name_str = 'UbuntuVM' #cn
    key1 = 'UbuntuVM' #key1
    event_category = envdict['Simulation']
    event_action = 'False' if envdict["Parameters_Var"] == None else 'True'
    event_label = "{}_{}".format(envdict['ncpus'],envdict['mpi_nbcpu'])
    tracking_url = 'https://www.google-analytics.com/collect?v=1&t=event&tid='+tracking_id+'&cid='+clientid_str+'&ec='+event_category+'&ea='+event_action+'&el='+event_label+'&key1='+key1+'&aip=0'
    requests.post(tracking_url)
    # Useful urls
    # https://www.optimizesmart.com/understanding-universal-analytics-measurement-protocol/
    # https://requests.readthedocs.io/en/master/user/quickstart/
    # https://developers.google.com/analytics/devguides/collection/protocol/v1/reference
