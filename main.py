import vectordb
import json
import time
from openai import OpenAI

openai = OpenAI()

pokemons = []
descriptions = []
ids = []
descriptions_embeddings = []
queries = [ "plays with thunder"]

with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        pokemon = json.loads(line)
        pokemons.append(pokemon)
        descriptions.append(pokemon["info"]["description"])
        ids.append(pokemon["info"]["id"])

def find_pokemon(id):
    for pokemon in pokemons:
        if (str(pokemon["info"]["id"]) == id):
            print(format(pokemon))
            break

def format(pokemon):
    return f"""Name: {pokemon["name"]}
Pokedex ID: {pokemon["info"]["id"]}
Type: {pokemon["info"]["type"]}
Description: {pokemon["info"]["description"]}
"""

def get_embeddings(descriptions: list[str]):
    embeddings = openai.embeddings.create(
        model="text-embedding-3-small",
        input=descriptions,
    )
    return [data.embedding for data in embeddings.data]

descriptions_embeddings = get_embeddings(descriptions)
query_embeddings = get_embeddings(queries)

client = vectordb.Client()

client.create_index(
    "pokemon",
    dimension=1536,
    allow_replace_deleted=True,
)

# Insert documents to index
t0 = time.time()
client.add("pokemon", ids, documents=descriptions, embeddings=descriptions_embeddings)
t1 = time.time()

print("Indexed", t1 - t0)

# Query with a text input
res = client.query(
    index="pokemon",
    query_embeddings=query_embeddings,
    k=3,
    include=["document", "metadata"]
)
t2 = time.time()

print("Searched", t2 - t1)

if res:
    for pokes in res:
        for p in pokes:
            find_pokemon(p["id"])
