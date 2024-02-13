import pyspark.sql.functions as sf
from tweet_json_explorer import df_tweet
from SVV_message_explorer import df

# %%
tweet_ids = [1579429693274853377,
             1579390029402902529,
             1579379449925373954,
             1579803250299392001,
             1579799343644958721,
             1579795847239593986,
             1579794832322867201,
             1579787644778995712,
             1579785075826503681,
             1579776407529721856,
             1579771779417243655,
             1580151163361849344,
             1580114486308241408,
             1392383430688714752,
             1392407870294069248,
             1392415965254758400,
             1392777524506251268,
             1393109116634157059,
             1393128673218338817,
             1393140739501080578,
             1393466028533993472,
             1393511893826932743,
             1393534753245171713,
             1394572452026531841,
             1394581466072371200,
             1394623613484118021,
             1394910880555577352,
             1394956466054967301,
             1395299169284284417,
             1395310686222364673,
             1395318462428368897,
             1395336442847576066,
             1369564918639394816,
             1369566939991924736,
             1369607426006220800,
             1369611365145534464,
             1369963175110717442,
             1369936431297232896,
             1369937633565155329,
             1369942447913074688,
             1369970899831644160,
             1369976376242098182,
             1370284490107592707,
             1370314739134398465,
             1370297724675952645,
             1370319398842421250,
             1370332107202035713,
             1370338123817889793,
             1370351535608242182,
             1370355647984832521,
             1370650912348323847,
             1370670566504591362,
             1370672139473465348,
             1371372851459162113,
             1371408838268743681,
             1371736896716554240,
             1371747263064764416,
             1371757809726001159,
             1371782727079460864,
             1372123698145157125,
             1372157899837997056,
             1372164631226163202,
             1372499910172508169,
             1372501127149842432,
             1372530210755727362,
             1372830491238367233,
             1372861087796854785,
             1373228475008622592,
             1373241803156099079,
             1373246278038130688,
             ]

svv_recordIds = [
    "f859e2e0-b192-4a29-896a-6790c161a816_2",
    "1b3ebabc-d52a-45ff-9851-f20879d0cb60_2",
    "47dd55b2-be1a-45b5-9f32-c59dee38c3d3_1",
    "050e7481-d649-445e-89e6-c4b77ae3816d_1",
    "75e7d320-2cd0-48d2-8dd0-7f2fd8b8c716_1",
    "f9632a1b-9a37-44ab-a64e-4d39577f6971_1",
    "c9e62f41-e7b3-47d6-8985-8357eb4200dd_1",
    "8028009e-4c79-43a5-89a0-3fe9074c1396_2",
    "b807f3f9-5964-4cf6-8788-5a24f117facd_2",
    "1be42be1-08f7-4d7e-a9fa-8f194af70461_1",
    "90c1e706-daa1-4327-82ff-583241c0eb77_1",
    "b7ee977c-0b94-418e-bcc7-e7c45ec55a32_1",
    "bdf91359-cd32-44dd-895a-52e37458d600_1",
    "NPRA_VL_295661_1",
    "NPRA_VL_295701_1",
    "NPRA_VL_295704_1",
    "NPRA_VL_295809_2",
    "NPRA_VL_295848_2",
    "NPRA_VL_295850_1",
    "NPRA_VL_295866_1",
    "NPRA_VL_295967_2",
    "NPRA_VL_295971_2",
    "NPRA_VL_295972_1",
    "NPRA_VL_296078_1",
    "NPRA_VL_296083_1",
    "NPRA_VL_296113_2",
    "NPRA_VL_296189_1",
    "NPRA_VL_296215_2",
    "NPRA_VL_296333_2",
    "NPRA_VL_296341_2",
    "NPRA_VL_278646_1",
    "NPRA_VL_296348_2",
    "NPRA_VL_290110_1",
    "NPRA_VL_290114_1",
    "NPRA_VL_290124_2",
    "NPRA_VL_290126_2",
    "NPRA_VL_290176_1",
    "NPRA_VL_290264_1",
    "NPRA_VL_290259_1",
    "NPRA_VL_290269_2",
    "NPRA_VL_290278_2",
    "NPRA_VL_290282_1",
    "NPRA_VL_290399_1",
    "NPRA_VL_290405_1",
    "NPRA_VL_290225_1",
    "NPRA_VL_290400_1",
    "NPRA_VL_290423_1",
    "NPRA_VL_290426_1",
    "NPRA_VL_290438_1",
    "NPRA_VL_290442_2",
    "NPRA_VL_290536_2",
    "NPRA_VL_290539_2",
    "NPRA_VL_290538_1",
    "NPRA_VL_290614_1",
    "NPRA_VL_290616_1",
    "NPRA_VL_290720_1",
    "NPRA_VL_290732_2",
    "NPRA_VL_290735_1",
    "NPRA_VL_290742_2",
    "NPRA_VL_290721_1",
    "NPRA_VL_290864_1",
    "NPRA_VL_290843_2",
    "NPRA_VL_290957_2",
    "NPRA_VL_290978_2",
    "NPRA_VL_290997_2",
    "NPRA_VL_291084_1",
    "NPRA_VL_291074_1",
    "NPRA_VL_291218_1",
    "NPRA_VL_291241_1",
    "NPRA_VL_291050_2"
]

svv_situation_ids = ["NPRA_HBT_10-10-2022.39293",
                     "NPRA_HBT_10-10-2022.39209",
                     "NPRA_HBT_10-10-2022.39182",
                     "NPRA_HBT_11-10-2022.39810",
                     "NPRA_HBT_11-10-2022.39738",
                     "NPRA_HBT_11-10-2022.39788",
                     "NPRA_HBT_11-10-2022.39794",
                     "NPRA_HBT_11-10-2022.39765",
                     "NPRA_HBT_11-10-2022.39766",
                     "NPRA_HBT_11-10-2022.39745",
                     "NPRA_HBT_11-10-2022.39732",
                     "NPRA_HBT_12-10-2022.40199",
                     "NPRA_HBT_12-10-2022.40142",
                     "NPRA_VL_295661",
                     "NPRA_VL_295701",
                     "NPRA_VL_295704",
                     "NPRA_VL_295809",
                     "NPRA_VL_295848",
                     "NPRA_VL_295850",
                     "NPRA_VL_295866",
                     "NPRA_VL_295967",
                     "NPRA_VL_295971",
                     "NPRA_VL_295972",
                     "NPRA_VL_296078",
                     "NPRA_VL_296083",
                     "NPRA_VL_296113",
                     "NPRA_VL_296189",
                     "NPRA_VL_296215",
                     "NPRA_VL_296333",
                     "NPRA_VL_296341",
                     "NPRA_VL_278646",
                     "NPRA_VL_296348",
                     "NPRA_VL_290110",
                     "NPRA_VL_290114",
                     "NPRA_VL_290124",
                     "NPRA_VL_290126",
                     "NPRA_VL_290176",
                     "NPRA_VL_290264",
                     "NPRA_VL_290259",
                     "NPRA_VL_290269",
                     "NPRA_VL_290278",
                     "NPRA_VL_290282",
                     "NPRA_VL_290399",
                     "NPRA_VL_290405",
                     "NPRA_VL_290225",
                     "NPRA_VL_290400",
                     "NPRA_VL_290423",
                     "NPRA_VL_290426",
                     "NPRA_VL_290438",
                     "NPRA_VL_290442",
                     "NPRA_VL_290536",
                     "NPRA_VL_290539",
                     "NPRA_VL_290538",
                     "NPRA_VL_290614",
                     "NPRA_VL_290616",
                     "NPRA_VL_290720",
                     "NPRA_VL_290732",
                     "NPRA_VL_290735",
                     "NPRA_VL_290742",
                     "NPRA_VL_290721",
                     "NPRA_VL_290864",
                     "NPRA_VL_290843",
                     "NPRA_VL_290957",
                     "NPRA_VL_290978",
                     "NPRA_VL_290997",
                     "NPRA_VL_291084",
                     "NPRA_VL_291074",
                     "NPRA_VL_291218",
                     "NPRA_VL_291241",
                     "NPRA_VL_291050"]

# %%
def find_situation_id_from_recordId(id):
    return df.select("situationId", sf.explode("recordId")).where(sf.col("col") == id).dropDuplicates(["situationId"])


def find_recordIds_from_situationId(id):
    return df.select("situationId", "recordId").where(sf.col("situationId") == id)


def find_situation_ids_from_list():
    situation_ids = df.join(
        df.select("situationId", sf.explode("recordId")).where(sf.col("col").isin(svv_recordIds)).dropDuplicates(
            ["col"]).select(
            "situationId"),
        df.select("situationId", sf.explode("recordId")).where(sf.col("col").isin(svv_recordIds)).dropDuplicates(
            ["col"]).select(
            "situationId").situationId == df.situationId).rdd.map(lambda x: x["situationId"]).collect()


# %%
find_situation_id_from_recordId("NPRA_VL_291050_2").show()

# %%
find_recordIds_from_situationId("NPRA_HBT_12-10-2022.40142").show(truncate=False)


# %%
def find_situation_ids_ordered_by_list():
    ids_list = []
    for id in svv_recordIds:
        ids_list.append(
            f'"{df.select("situationId", sf.explode("recordId")).where(sf.col("col") == id).dropDuplicates(["situationId"]).rdd.map(lambda x: x["situationId"]).collect()[0]}"')
    print(ids_list)


find_situation_ids_ordered_by_list()

#%%
for a,b in zip(svv_situation_ids, tweet_ids):
    print(df.where(sf.col("situationId") == a).rdd.map(lambda x: x["desc"] + x["dataProcessingNote"]).collect(),
          df_tweet.where(sf.col("id") == b).select("full_text").rdd.map(lambda x: x["full_text"]).collect())

#%%

for i in range(3):
    df.where(sf.col("situationId") == svv_situation_ids[i]).select("overallStartTime", "dataProcessingNote", sf.col("locations")[0]["locationDescriptor"]).show(truncate=False, vertical=True)
    df_tweet.where(sf.col("id") == tweet_ids[i]).select("created_at", "full_text").show(truncate=False, vertical=True)

#%%