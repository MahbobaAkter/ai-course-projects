## Part 1 — Understanding the Demo Code

**1. State fields and node behavior**
The `State` contains three fields: `query`, `context`, and `answer`.
The `retrieve` node reads the user’s query and searches the vector database to find similar documents. It then stores those results in the `context`.
The `generate` node reads both the query and the retrieved context. It uses this information to generate a final answer using the language model and stores it in `answer`.

---

**2. Meaning of k=3**
The retriever is set with `k=3`, which means it returns the top 3 most similar documents for a query.
If we change it to `k=1`, the result becomes more precise but may miss useful information.
If we change it to `k=10`, it returns more documents, but some of them may be irrelevant, which can reduce the quality of the answer.

---

**3. Effect of removing SystemMessage**
The `SystemMessage` gives instructions to the model about how to answer.
When it is removed, the answers become less structured and less consistent. The model may also not follow the context strictly, so the responses can feel less clear or less focused.

---

**4. Retrieval quality observation**
When I checked the retrieved documents using the debug print, I noticed that the results are not always accurate.
For clear queries, the correct creatures are usually retrieved. However, for vague or unclear queries, the system sometimes returns unrelated creatures.
This shows that the quality of retrieval directly affects how good the final answer is.
