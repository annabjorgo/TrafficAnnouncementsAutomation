import torch
from sentence_transformers import SentenceTransformer, util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
#%%

input_texts = [
    "jeg heter mads og liker banan",
    "jeg heter anna men hater banan",
]

em1 = model.encode(input_texts[0], convert_to_tensor=True)
em2 = model.encode(input_texts[1], convert_to_tensor=True)
#%%
util.pytorch_cos_sim(em1, em2)

#%%
a_embeddings = model.encode(list(a['conc']), convert_to_tensor=True)
b_embeddings = model.encode(list(b['full_text']), convert_to_tensor=True)

#%%
r = util.pytorch_cos_sim(a_embeddings[0], b_embeddings)
print(r.argmax())
print(a.iloc[0])
print(b.iloc[int(r.argmax())])
#%%
correct_counter = 0
counter = 0
for index, embedding in enumerate(a_embeddings):
    print(counter)
    counter += 1
    sim = util.pytorch_cos_sim(embedding, b_embeddings)
    a_id = a.iloc[index]['recordId']
    b_id = b.iloc[int(sim.argmax())]['id']
    if str(merge_dict[a_id]).strip() == str(b_id).strip():
        correct_counter += 1
    else:
        print(f'Correct svv: {a.iloc[index]["conc"]}', "\n", f"Predicted tweet: {b.iloc[int(sim.argmax())]['full_text']}",
              "\n", f"Actual tweet: {b[b['id'] == str(merge_dict[a.iloc[index]['recordId']])].iloc[0]['full_text']}")

print(correct_counter / counter)

#%%