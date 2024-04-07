#%%
import sys
sys.path.append('TrafficAnnouncementsAutomation/data_explorations')
from SVV_message_explorer import df
import pyspark.sql.functions as sf

#%%

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
    "NPRA_VL_291050_2",
    "NPRA_VL_293723_1",
    "NPRA_VL_293707_1",
    "NPRA_VL_293692_1",
    "NPRA_VL_293691_1",
    "NPRA_VL_293745_1",
    "NPRA_VL_293740_1",
    "NPRA_VL_293737_1",
    "NPRA_VL_293736_1",
    "NPRA_VL_293733_2",
    "NPRA_VL_293728_1",
    "NPRA_VL_293726_1",
    "NPRA_VL_296338_1",
    "NPRA_VL_296351_1",
    "NPRA_VL_296360_1",
    "NPRA_VL_296362_1",
    "NPRA_VL_296365_1",
    "NPRA_VL_296375_1",
    "NPRA_VL_296380_2",
    "NPRA_VL_296393_1",
    "NPRA_VL_293376_1",
    "NPRA_VL_293785_2",
    "NPRA_VL_293866_2",
    "NPRA_VL_293872_2",
    "NPRA_VL_293853_2",
    "NPRA_VL_293884_1",
    "NPRA_VL_293902_1",
    "NPRA_VL_294048_1",
    "NPRA_VL_294046_1",
    "NPRA_VL_294108_1",
    "NPRA_VL_294107_1"
]

df_filtered = df.filter(sf.col("recordId").isin(svv_recordIds)).select(sf.col("description"), sf.col("dataProcessingNote"), sf.col("locationDescriptor"))
df_concat_location_description = df_filtered.withColumn(
    "text",
    sf.concat_ws(
        " ",
        sf.col("description"),
        sf.col("dataProcessingNote")
    )
).withColumn(
    "locationAsString",
    sf.array_join(sf.col("locationDescriptor"), ", ")
).select(sf.col("locationAsString"),sf.col("text"))

df_concat_location_description_pd =df_concat_location_description.toPandas()




#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#%%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# må vurdere true eller false på do_sample og temperature
def rephrase_svv_to_tweet(locations, sentences, batch_size=10, ):
    rephrased_sentences = []
    for i in range(0, len(sentences), batch_size):
        batch_locations = locations[i:i + batch_size]
        batch_texts = sentences[i:i + batch_size]
        prompts = [
            (f"Gitt stedet \"{location}\" og situasjonen \"{text}\", omformuler teksten for å gjøre den klarere og mer sammenhengende på norsk. Målet er å skrive setningen på formen lokasjon: hendelse.\n"
             )
            for location, text in zip(batch_locations, batch_texts)]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        batch_rephrased = [tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in generated_ids]
        rephrased_sentences.extend(batch_rephrased)
    return rephrased_sentences


rephrased = rephrase_svv_to_tweet(
    df_concat_location_description_pd['locationAsString'].tolist(),
    df_concat_location_description_pd['text'].tolist(),
    batch_size=10
)

# Assuming your DataFrame has a suitable column for storing the rephrased text
df_concat_location_description_pd['rephrased'] = rephrased


#%%
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

#%%

prompt_one = """Skriv om følgende tekst til en komplett, lesbar setning. Meldingen skal være en traffikkmelding som skal ut til det norske folk og det er viktig at all informasjon er korrekt. Meldingen skal være på formen veinummer+ region/strekning: melding. konsekvens.\n"
    f"Eksempel 1: E6 Oslo-Gardermoen: Stengt ved kryss 47 Grankrysset for trafikken fra Oslo mot Gardermoen grunnet trafikkulykke. Køen starter ved kryss 45 Skedmovollen.\n "
    f"Eksempel 2: Fv 60 Sunnmøre: Fra Stranda sentrum opp mot Strandafjellet har et vogntog problem. Feltet opp mot fjellet er sperret.\n "
    f"Eksempel 3: E6 Trondheim-Stjørdal: Helltunnelen stengt sørover mot Trondheim pga bilberging. Bilberger er på stedet.\n"
    f"Følgende tekst som skal skrives om er: '{location} {text}'"""

prompt_two = """Gitt stedet '{location}' og situasjonen '{text}', omformuler teksten for å gjøre den klarere og mer sammenhengende på norsk. 
Målet er å skrive en kort setning på formen lokasjon: hendelse. Den nye teksten er: """

prompt_three = """Skriv denne teksten om til en komplett tekst på formen [lokasjon : hendelse]: {location}, {text}. Den nye teksten er: """




#%%
def generate_reformulated_text(model_name, max_new_tokens=150, top_k=40, top_p=0.85, temperature=0.7,
                               do_sample=True, repetition_penalty=1.2, num_return_sequences=1):
    locations = df_concat_location_description_pd['locationAsString'].tolist()[:10]
    texts = df_concat_location_description_pd['text'].tolist()[:10]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)



    for i in range(len(texts)):
        prompt = prompt_three.format(location=locations[i], text=texts[i])
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = tokenizer(prompt, return_tensors='pt').attention_mask
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Input: {locations[i]}. {texts[i]}\nGenerated: {generated_text}\n")

#%%  NB-GPT-J
generate_reformulated_text("NbAiLab/nb-gpt-j-6B")

#%% Mistral
generate_reformulated_text("mistralai/Mistral-7B-Instruct-v0.2")

#%% mT5
from transformers import T5Tokenizer, MT5ForConditionalGeneration


#%%
model_name = "google/mt5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

input_text = "rephrase: Ev 6 [41] Furuset, på strekningen Gardermoen - Oslo (Alnabru), i retning mot Oslo (Alnabru), Åpen for trafikk."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)