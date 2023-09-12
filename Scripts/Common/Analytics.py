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
    tracking_id = 'UA-112907949-3' #tid
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

'''
# The following is the new method for google analytics.
# GA4 replaces UA which has been discontinued.
# Currently, the variables in the script below have been hard-coded.
# These need amending to match the behaviour of the script above.

# Variables to be amended:
# client_id, timestamp_micros, name, items{"name_1":"string_name1"}, params
# items are embedded within params. Use the guided event builder (link below)
# to figure best way to organise these

import requests
import json
url = "https://www.google-analytics.com/mp/collect?measurement_id=G-M4X1J02VS1&api_secret=XwWir8mZRzees0BOKtHFmg"
payload = {
  "client_id": "manual",
  "timestamp_micros":"1694535623258000",
  "non_personalized_ads": True,
  "events": [
    {
      "name": "Label",
      "params": {
        "items": [
            {"name_1":"string_name1"}
        ],
        "key1": "UbuntuVM"
      }
    }
  ]
}
r = requests.post(url,data=json.dumps(payload),verify=True)
print(r.status_code)

# Useful urls
# https://codelabs.developers.google.com/codelabs/GA4_MP#0 # Tutorial for getting this to work
# https://ga-dev-tools.google/ga4/event-builder/ # Guided event builder
# https://colab.research.google.com/ # Online tool for testing
'''