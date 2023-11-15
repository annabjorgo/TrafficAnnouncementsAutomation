from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

#%%
import json
import xmltodict


with open("situation.xml") as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
    # xml_file.close()

    # generate the object using json.dumps()
    # corresponding to json data

    json_data = json.dumps(data_dict)

    # Write the json data to output
    # json file

#%%
with open("../data/xml_data.json", "w") as json_file:
    json_file.write(json_data)
    json_file.close()


#%%
spark = SparkSession.builder.appName("ReadXML").getOrCreate()

xmlFile = "/Users/madslun/Documents/Fag-Programmering/TrafficAnnouncementsAutomation/data.json"

df = spark.read \
    .options(rowTag='root') \

