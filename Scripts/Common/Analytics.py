import requests
from datetime import datetime
import inspect
import os
import ast

'''
This function serves to provide a small amount of analytics data on how
VirtualLab is used. VirtualLab is open-source software which has been developed
through the support of research grants. The small amount of statistics this
function gathers enables us to evidence the impact and value of our research.
This is invaluable when we apply for funding which will lead to further
development of VirtualLab for your benefit.

The anonymised data which is sent to us is the following: Number of jobs run,
and number of those jobs run in parallel.

We hope you agree to this data being sent to us to assist us in applying for
future research grants. If you would like to disable this feature, this may be
done in VLconfig.py.
'''

def Run(Category,Action,Label):
    #tracking_id = 'UA-112907949-3' #tid
    tracking_id = 'G-M4X1J02VS1' #tid
    clientid_str = str(datetime.now()) #cid
    campaign_name_str = 'UbuntuVM' #cn
    key1 = 'UbuntuVM' #key1

#    tracking_url = "https://www.google-analytics.com/collect?v=1&t=event&tid={}&cid={}\
#                    &ec={}&ea={}&el={}&key1={}&aip=0".format(tracking_id,clientid_str,
#                                        Category,Action,Label,key1)

    tracking_url = "https://www.google-analytics.com/collect?v=2&t=event&tid={}&cid={}\
                    &ec={}&ea={}&el={}&key1={}&aip=0".format(tracking_id,clientid_str,
                    Category,Action,Label,key1)

    try:
        requests.post(tracking_url)
    except :
        pass
    # Useful urls
    # https://www.themarketingtechnologist.co/measure-your-python-projects-with-google-analytics/
    # https://www.optimizesmart.com/understanding-universal-analytics-measurement-protocol/
    # https://requests.readthedocs.io/en/master/user/quickstart/
    # https://developers.google.com/analytics/devguides/collection/protocol/v1/reference
