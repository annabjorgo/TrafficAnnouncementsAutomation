import requests
import sklearn.metrics.pairwise as cos_sim

API_URL = "https://api-inference.huggingface.co/models/princeton-nlp/sup-simcse-roberta-large"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


output1 = query({
    "inputs": "E6 Skedsmokorset: Bil står i venstre fil retning Gardermoen. Skaper Kø. Vær OBS.",
})

output2 = query({
    "inputs": "E6 Fåberg i Lillehammer-Ensby i Øyer vil i kveld bli stengt i inntil 20 minutter mellom kl. 18.30 og 19:30 pga. sprengningsarbeid."
})
print(output1, output2)

print(cos_sim.cosine_similarity(output1, output2))