Fine tuning is a process used in machine learning to improve a pre-trained model on a specific task or domain. It's a form of what's called [[Transfer Learning]], where a model's existing knowledge learned from a large, general dataset is used as a starting point and then adapted using new, task-specific training data.

A **pre-trained** model is one which has already been trained on large-scale datasets (e.g. Common Crawl, Wikipedia, ImageNet) by someone else, and has been made available for others to use. Examples of pre-trained model's could be **GPT-4o** and **Gemini 1.5 Flash**. Both these models can be fine-tuned via their respective APIs and SDKs.

Fine-tuning typically involves updating some or all of the model’s parameters using a smaller, focused dataset.

## Benefits

- Allows for domain specific knowledge and improved performance on specific tasks compared to training models from scratch.
- **Resource-efficient**: avoids the need to train a model from scratch, which can be extremely expensive in both the data needed and computational resources.
- Great for correcting hallucinations that are proving difficult to fix through traditional prompt engineering techniques.
- Can make models better at following instructions via instruction tuning or [[RLHF]].

## Example Use Cases

- Natural Language Processing (NLP) - training an LLM to perform sentiment analysis tasks.
- Computer Vision (CV) - adapting a model used for object detection to identify new objects it's never seen before.

## When NOT to Use Fine-Tuning

Fine-Tuning is great, but it's not always the best or even the most efficient solution. In most cases, lighter alternatives should always be attempted first such as prompt engineering, few shot learning, or RAG.

## Where Fine-Tuning Makes Sense

| ✅ Criteria                                                          | Why it Matters                                                                                                             |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **You need consistent, high-quality output across many requests**    | Prompt-based methods may vary from run to run. Fine-tuning helps eliminate this variability.                               |
| **Prompting fails to fix model behavior**                            | If the model keeps hallucinating or misunderstanding instructions, fine-tuning lets you embed the correct behavior deeply. |
| **You want low-latency deployment**                                  | Fine-tuned models (especially [[Quantised Models\|quantised]] ones) can run faster than complex RAG systems.               |
| **You're deploying in a product with strict UX or compliance needs** | Prompt engineering isn’t stable enough for enterprise-grade use cases in many situations.                                  |

## Fine-Tuning Techniques

### 🧠 Full vs. Parameter-Efficient

- [[Full Fine-Tuning]]
- [[PEFT|Parameter-Efficient Fine-Tuning (PEFT)]]
  - [[Low Rank Adaptation (LoRA)]]
  - [[QLoRA]]

### ✍️ Prompt-Based Tuning

- [[Instruction Tuning]]
- [[Prompt Tuning]]
- [[Prefix Tuning]]
- [[P-Tuning]] (Not as commonly used)

### 🔌 Modular Approaches

- [[Adapter Methods]]

### ⚖️ Alignment & Preference Optimization

- [[RLHF|Reinforcement Learning from Human Feedback (RLHF)]]
  - [[DPO|Direct Preference Optimization (DPO)]]

> [!Tip]
> Overwhelmed? Not sure where to start? Here's a quick guide to the most common fine-tuning techniques, what they do, and when you'd use them.

## 🧩 Fine-Tuning Techniques — Summary & Comparison

### 🔹 **PEFT (Parameter-Efficient Fine-Tuning)**

This should be where you start if fine-tuning is the route you want. QLoRA/LoRA specifically.

- **What it is:** Only a _small subset_ of the models parameters are trained.
- **When to use it:** You want to fine-tune on limited data or budget.
- **Includes techniques like:**
	- **[[Low Rank Adaptation (LoRA)]]:** Injects low-rank matrices into model layers.
	- **[[QLoRA]]:** A more efficient version that uses quantized weights for even smaller resource use.
- **Example use case:** Adapting a general LLM to answer finance-specific questions with a small dataset.

### 🔹 Full Fine-Tuning

- **What it is:** Updating all the model's parameters.
- **When to use it:** You have lots of data, strong compute resources, and want maximum flexibility/customisation.
- **Downsides:** Very expensive and resource intensive :(
- **Example use case:** Training a specialized biomedical model from GPT-like architecture.

