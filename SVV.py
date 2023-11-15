import requests

headers = {
    'accept': 'application/vnd.svv.v1+json;charset=utf-8',
    'X-System-ID': 'SYSTEM-KORTNAVN',
}

#%%
response = requests.get('https://traffic-info.atlas.vegvesen.no/traffic-information/messages', headers=headers)

#%%
response2 = requests.get('https://traffic-info.atlas.vegvesen.no/traffic-information/messages/NPRA_HBT_09-10-2023.139605', headers=headers)

#%%
a  =0
sorted()