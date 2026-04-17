import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings


# ─── Fictionary Creature Catalog ───────────────────────────────────────────

CREATURES = [
    {
        "name": "Gloomfang",
        "type": "Shadow Beast",
        "habitat": "Dark Forests",
        "size": "Large",
        "abilities": ["Shadow Step", "Fear Aura", "Night Vision"],
        "diet": "Carnivore",
        "danger_level": 8,
        "description": "A wolf-like creature made of living shadow. Its fur absorbs light, making it nearly invisible at night.",
    },
    {
        "name": "Crystalwing",
        "type": "Aerial Elemental",
        "habitat": "Mountain Peaks",
        "size": "Medium",
        "abilities": ["Ice Breath", "Crystal Shield", "Blizzard Call"],
        "diet": "Omnivore",
        "danger_level": 6,
        "description": "A bird with wings made of translucent ice crystals. It soars at high altitudes and can summon blizzards.",
    },
    {
        "name": "Murkwraith",
        "type": "Swamp Spirit",
        "habitat": "Marshes and Bogs",
        "size": "Small",
        "abilities": ["Poison Mist", "Bog Sink", "Mimic Voice"],
        "diet": "Soul Eater",
        "danger_level": 7,
        "description": "A translucent spirit that floats above swamp water, luring travelers with mimicked voices before dragging them under.",
    },
    {
        "name": "Emberclaw",
        "type": "Fire Drake",
        "habitat": "Volcanic Regions",
        "size": "Huge",
        "abilities": ["Magma Breath", "Heat Aura", "Armor Melt"],
        "diet": "Carnivore",
        "danger_level": 9,
        "description": "A small dragon variant with claws that glow like molten rock. It can melt metal armor on contact.",
    },
    {
        "name": "Thornback",
        "type": "Forest Armored",
        "habitat": "Ancient Woodlands",
        "size": "Large",
        "abilities": ["Thorn Volley", "Bark Armor", "Root Grasp"],
        "diet": "Herbivore",
        "danger_level": 4,
        "description": "A tortoise-like creature covered in living thorns. Despite being herbivorous, it aggressively defends its territory.",
    },
    {
        "name": "Voidwhisper",
        "type": "Psychic Specter",
        "habitat": "Abandoned Ruins",
        "size": "Incorporeal",
        "abilities": ["Mind Read", "Memory Steal", "Illusion Cast"],
        "diet": "Memory Eater",
        "danger_level": 8,
        "description": "An invisible entity that feeds on memories. Victims often wake with no recollection of their past.",
    },
    {
        "name": "Saltmaw",
        "type": "Sea Lurker",
        "habitat": "Coastal Waters",
        "size": "Gigantic",
        "abilities": ["Tidal Pull", "Brine Spit", "Echo Roar"],
        "diet": "Piscivore",
        "danger_level": 7,
        "description": "A massive eel-like creature with rows of bioluminescent teeth, known for capsizing fishing boats.",
    },
    {
        "name": "Duskmorel",
        "type": "Fungal Wanderer",
        "habitat": "Underground Caves",
        "size": "Medium",
        "abilities": ["Spore Cloud", "Mycelium Network", "Regenerate"],
        "diet": "Decomposer",
        "danger_level": 3,
        "description": "A walking mushroom colony that releases hallucinogenic spores when threatened. Mostly harmless unless cornered.",
    },
]


# ─── ChromaDB vector store (in-memory, built at startup) ─────────────────────


embeddings = FakeEmbeddings(size=768)

vector_store = Chroma.from_texts(
    texts=[json.dumps(c) for c in CREATURES],
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ─── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    query: str
    context: list[str]
    answer: str
    sources: list[str]
    retry_count: int

# ─── LLM ─────────────────────────────────────────────────────────────────────

#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


# ─── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["query"])

    context = []
    sources = []

    for doc in docs:
        context.append(doc.page_content)

        try:
            data = json.loads(doc.page_content)
            sources.append(data.get("name", "unknown"))
        except:
            sources.append("unknown")

    count = state.get("retry_count", 0) + 1

    return {
        "context": context,
        "sources": sources,
        "retry_count": count
    }

def format_answer(state: State) -> dict:
    answer = state["answer"]

    sentences = answer.split(". ")

    formatted = "\n".join(
        [f"{i+1}. {s.strip()}" for i, s in enumerate(sentences) if s]
    )

    return {"answer": formatted}


def generate(state: State) -> dict:
    context = state["context"]

    answer = "Based on the retrieved creatures:\n"

    for c in context:
        try:
            data = json.loads(c)
            answer += f"- {data['name']}: {data['description']}\n"
        except:
            pass

    return {"answer": answer}
    

   

# ─── Graph ────────────────────────────────────────────────────────────────────
#
#   START → retrieve → generate → END
#
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("format_answer", format_answer)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "format_answer")
builder.add_edge("format_answer", END)

graph = builder.compile()


# ─── Demo queries ─────────────────────────────────────────────────────────────

queries = [
    "What creatures live in dark or shadowy environments?",
    "Which creature is the most dangerous and what are its abilities?",
    "Tell me about the Murkwraith",
]

for query in queries:
    print(f"\n{'=' * 60}")
    print(f"Query:  {query}")
    print("-" * 60)

    result = graph.invoke({
        "query": query,
        "retry_count": 0
    })

    print(f"Answer:\n{result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Retries: {result['retry_count']}")