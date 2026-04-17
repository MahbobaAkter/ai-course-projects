# Week 5 - LangGraph

## Overview

In this assignment, I implemented a basic RAG (Retrieval-Augmented Generation) pipeline using LangGraph. The system retrieves relevant documents, generates an answer using an LLM, and extends the workflow with additional nodes and state tracking.

---

## Part 1 — Understanding the Demo Code

**1. State fields and node behavior**
The state contains three main fields: `query`, `context`, and `answer`.
The `retrieve` node reads the query and performs a similarity search to fetch relevant documents, which are stored in the context.
The `generate` node reads both the query and the context and uses them to produce the final answer.

---

**2. Effect of search_kwargs={"k": 3}**
The parameter `k=3` means that the retriever returns the top 3 most similar documents.
If `k=1`, the result is more precise but may miss important information.
If `k=10`, more documents are retrieved, but this can introduce irrelevant or noisy information, which may reduce answer quality.

---

**3. Removing the SystemMessage**
The SystemMessage provides instructions to the LLM about how to behave.
When it is removed, the answers become less structured and less consistent in tone. The responses may also be less focused on the provided context.

---

**4. Observations from retrieved documents**
The retrieved documents are not always correct.
For vague or unclear queries, the retriever may return irrelevant or unrelated documents. This shows that retrieval quality directly affects the final answer.

---

## Part 2 — Graph Extensions

In this part, I extended the original LangGraph pipeline with additional state fields and nodes.

* Added a new field `sources` to store the names of retrieved documents
* Added a new node `format_answer` to format the answer as a numbered list using Python string operations
* Added a `retry_count` field to track how many times the retrieval node is executed

These changes demonstrate how LangGraph allows flexible state updates and the addition of non-LLM processing nodes.

---

## Part 3 — Retrieval Quality Exploration

**Observations from experiments**
Well-structured queries usually return relevant results, while vague or misspelled queries reduce retrieval accuracy.
This leads to lower-quality answers because the model depends heavily on the retrieved context.

---

**Part 3c — Relevance Function**

```python
def is_relevant(query: str, documents: list) -> bool:
    query_words = set(query.lower().split())

    for doc in documents:
        text = doc.lower()
        matches = sum(1 for word in query_words if word in text)

        if matches >= 2:
            return True

    return False
```

This function checks whether there is sufficient keyword overlap between the query and the documents. It provides a simple heuristic for determining relevance.

---

## Part 4 — Conceptual Understanding

**1. Behavior with irrelevant retrieval**
If the retrieved documents are not relevant, the system still generates an answer. However, the answer may be incorrect or hallucinated because it is based on poor context.

---

**2. Improved graph design with relevance checking**
A better design would include a relevance-checking node after retrieval.
If the documents are not relevant, the system should retry retrieval or modify the query.
If they are relevant, it proceeds to the generation step.

---

**3. Reranking vs Filtering**
Reranking means ordering documents based on their relevance score.
Filtering means removing documents that do not meet a certain relevance threshold.
Reranking is useful when all documents are somewhat relevant, while filtering is better when many irrelevant documents are present.

---
