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

---

##Part 2 – Extending the LangGraph

In this part, I modified the original Demo 4 LangGraph code by adding new fields and nodes to improve functionality and demonstrate how state flows through the graph.

## 2a. Added sources field to State

I extended the State TypedDict by adding a new field called sources, which stores a list of creature names retrieved from the vector database.

In the retrieve node, I parsed each retrieved document (which is in JSON format) and extracted the "name" field. These names were then stored in the sources list and returned as part of the updated state.

At the end of the program, I printed the list of sources alongside the answer. This helps to understand which documents were used to generate the response.

## 2b. Added format_answer node

I created a new node called format_answer and placed it between the generate node and the END node in the graph.

This node takes the generated answer string and reformats it into a numbered list using simple Python string manipulation (splitting sentences and adding numbering).

This demonstrates that not all nodes in LangGraph need to use an LLM — some tasks can be handled using regular Python logic.

## 2c. Added retry_count field

I added another field called retry_count to the State, with an initial value of 0.

Inside the retrieve node, I incremented this value each time the node is called. This allows tracking how many times retrieval happens.

At the end of execution, I printed the final retry count. This prepares the system for future improvements, such as retry loops and conditional logic in later exercises.

##Summary

Through these modifications, I learned how to:

Extend the state with additional fields
Pass and update data across multiple nodes
Add new nodes to the graph flow
Combine LLM-based and non-LLM logic in a pipeline