> [!Tip]
> A lot of the initial high-level information in these notes come from a great video by Edward Hu, the guy who initially led the research into the invention of LoRA: https://www.youtube.com/watch?v=DhRoTONcyZE. Would highly recommend a watch before reading this.
>
> This video is also pretty good for a more manageable explanation: https://www.youtube.com/watch?v=t509sv5MT0w
>
> Don't worry about the maths and harder concepts of it too much, I'll explain it in one place here.

---

## Introduction to LoRA

LoRA (Low-Rank Adaptation) is a technique that makes fine-tuning large language models (LLMs) significantly more efficient and cost-effective. It's designed to reduce the compute, memory, and storage demands typically required to adapt these models to new tasks.

Announced in a [2021 paper](https://arxiv.org/pdf/2106.09685) by Microsoft Research, LoRA has become one of the most popular methods for fine-tuning LLMs due to the reduced need for enormous compute resources.

## What problem does it solve?

Imagine you have a powerful LLM such as GPT-4o that's been trained on massive amounts of general knowledge, but you want to make it better at a specific task such as answering questions on internal documentation or internal company processes.

Traditionally, you'd need to go through the long process of:

1. **Download the entire model** - These models can be tens to hundreds of gigabytes in size depending on the [[Precision in LLMs|precision]] (e.g., FP32 vs 4-bit quantized).
2. **Update all internal parameters** - This means adjusting billions of weights and biases - these are the actual numbers (parameters) that are stored in the connections between neurons (the lines in neural net diagrams).
3. **Store the new updated copy of the model** - Again, sometimes hundreds of gigabytes.

This process requires expensive high-end GPUs (such as Nvidia A100s), lots of memory, and a significant amount of storage space - causing it to be out of reach for many developers and small to medium sized teams.

## How LoRA solves that problem

LoRA takes a clever shortcut: instead of updating the entire model, it trains a pair of much smaller “adapter” matrices that represent the necessary task-specific changes.

These adapters:

- Are tiny compared to the full model (often <1% of the size).
- Can be trained on consumer grade hardware.
- Can be swapped in and out programmatically depending on the knowledge the model requires.

Think of LoRA like adding small, specialised compartments of knowledge into a larger, more powerful general-purpose model. You get to keep all the original capabilities, while adding your own customisations.

You can even combine or merge multiple LoRA adapters for different tasks.

## Why is this useful to front end developers such as us?

- Allows us to experiment with fine-tuning without having deep specialised ML expertise.
- We can create domain-specific models tailored to your app, product, or company data.
- We don’t need to modify the full base model — just train and plug in lightweight adapters.
- Adapters can be small enough to run entirely inside the browser without the need for servers via tools such as [WebLLM](https://github.com/mlc-ai/web-llm).

In the following sections, I'll explain how LoRA does this through some clever maths - but don't worry, each step will be broken down step by step.

## How to get started with LoRA

### No code solutions

For non-devs or people who prefer a no-code approach, several tools have emerged (ranked by PoC (Proof of Concept)/production readiness):

1. **[Hugging Face AutoTrain](https://huggingface.co/autotrain)**: Most production-ready with enterprise support, seamless model hosting, and integration with the Hugging Face ecosystem.
2. **[Predibase](https://predibase.com/)**: Purpose-built for enterprise use with robust monitoring, versioning, and deployment features - built with [[MLOps]] in mind.
3. **[LM Studio](https://lmstudio.ai/)**: Desktop application with built-in LoRA fine-tuning capabilities, best for early experimentation.
4. **[Trainy.ai](https://trainy.ai/)**: User-friendly but newer platform, better suited for initial exploration than production.

### Low-Code Solutions

- **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: Command-line tool with YAML configuration that offers excellent customization and reproducibility for more serious projects. If you're interested in Axolotl, examples can be found here: [Setup Examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples).
- **[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)**: GUI-based tool with solid features and active development, good for advanced PoCs. Despite the name, many models are supported, not just Llama (e.g., Mistral, Qwen, etc.).

### Developer-Focused Solutions

> [!Danger]
> Unfortunately the JavaScript LoRA ecosystem for training is still severely limited. Python is best for now. I will try to include JS alternatives for training where possible.
>
> However, inferencing LoRA-adapted models is still very much possible in the JS ecosystem. I recommend training in Python and inferencing in JS when needed.

For developers wanting more **_fine-tuned_** control over the process, these tools are what you want:

1. **Hugging Face Transformers + PEFT** (Python) - This is probably the most popular ecosystem for LoRA and has excellent [documentation](https://huggingface.co/docs/peft/index) on community support. PEFT also supports alternate techniques such as QLoRA, AdaLoRA and IA3.

   - **QLoRA** – Quantized LoRA, enabling training on 4-bit models.
   - **AdaLoRA** – Dynamically adjusts the size of LoRA adapters during training to save compute.
   - **IA3** – A simpler parameter-efficient fine-tuning method that's faster to train.

2. **Accelerate + DeepSpeed + PEFT**
   - **[Accelerate](https://huggingface.co/docs/accelerate/index)**: Simplifies distributed training across multiple GPUs/nodes. Used under the hood by tools such as **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**.
   - **[DeepSpeed](https://www.deepspeed.ai/)**: A deep learning optimisation library created by Microsoft designed to train large-scale models with billions of parameters faster, with less memory, and on cheaper hardware.
   - Useful for teams who want to train larger models on limited hardware via memory optimisation.

## Cloud Training Platforms

These are great for if you don't have the hardware required to train models locally.

1. **[Lightning AI](https://lightning.ai/)**: Uses an embedded version of VS Code and has a somewhat generous free tier. Has functionality for deploying and hosting of fine-tuned models.

2. **[Google Colab](https://colab.research.google.com/)**: Not strictly only for ML work but provides limited access to free 16GB VRAM GPUs, good enough for small model fine tuning. Pro version is slightly cheaper than Lightning AI's pro version.

### Inferencing tools

1. **[transformers.js](https://github.com/xenova/transformers.js)** – Hugging Face models, inference only. While it doesn’t support LoRA directly, you can merge LoRA weights into a base model before exporting (via `peft_model.merge_and_unload()` from the Python `peft` library before saving with `transformers`).

2. **[ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web)** – Load exported [[ONNX]] models with merged LoRA weights.

## The theory of how LoRA works

LoRA is expressed using the equation:

$$
W + \Delta W = W + BA
$$

Where B and A are new 'low-rank matrices', also known as our adapters.

To easily understand what that equation is doing, let's break it down into a step by step process.

1. The first step is to calculate what we call 'delta W (ΔW)'. This can be expressed as:
   $$
   \Delta W=BA
   $$

So where do we start here? First we need to figure out what the `B` and `A` variables are. Within the concept of LoRA, `B` and `A` are what we call 'low-rank matrices'. Imagine we have a frozen weight — a matrix of parameters in the model that we don’t change during fine-tuning. (In a neural network diagram, these are the lines connecting neurons). This weight contains a matrix with dimensions of `4096 rows x 11008 columns`.

We need to establish a value for what we call the 'rank'. In LoRA, the **rank** is a number that controls how much "customization" we're adding to the model. A lower rank means smaller, faster fine-tuning — but it might not capture complex changes. A higher rank adds more detail but costs more to train. A typical starting point would be something like `rank = 8`.

The "rank" determines the size of our two low-rank matrices. We build:

- `B` with shape `[4096, r]`
- `A` with shape `[r, 11008]`

These two smaller matrices are then multiplied to create `ΔW`, which has the same shape as the original weight matrix `[4096 x 11008]`.

For example, the calculation would look like this:

$$
\begin{aligned}
B &= 4096 \times r \\\\
A &= r \times 11008
\end{aligned}
$$

Where `r` represents our rank value. To visualize the shape of the matrix multiplication, think of it like:

$$
\Delta W = [4096 \times r] \cdot [r \times 11008]
$$

This results in a matrix of shape `[4096 x 11008]`, exactly matching the shape of the original weight matrix `W`. This equation tells us what the value of 'delta W' is. 'delta W' is just a variable we can use to represent the changed version of the original frozen weight. It's worth remembering that during fine-tuning, we _only_ update the low-rank matrices `A` and `B`. The original weight matrix `W` stays frozen, saving both memory and compute.

2. So now we know what the value of 'delta W' is, we can go back to our original equation:

$$
W + \Delta W = W + BA
$$

What we need to do next is to perform a matrix addition between the original weight (`W`) and 'delta W'. This is represented as:

$$
W + \Delta W
$$

That equation represents the final **_fine tuned_** weight that can be inferenced on. To inference on that newly trained weight it'd look something like this:

$$
(W + \Delta W) \times input
$$

Where **_input_** is some form of embedded text that you might send to the LLM. You'll then get a response from your newly fine tuned model with the new data you trained it on.

### JavaScript Code Example

```JavaScript
import { Matrix } from 'ml-matrix';

// Step 1: Define base (frozen) weight matrix dimensions
const numRows = 4096;
const numCols = 11008;
const rank = 8;

// Step 2: Simulate the frozen weight matrix W (not modified during fine-tuning)
const frozenWeight = Matrix.rand(numRows, numCols); // values between 0 and 1

// Step 3: Create low-rank matrices B and A
// B: shape [4096, 8]
// A: shape [8, 11008]
const B = Matrix.rand(numRows, rank).mul(0.01);
const A = Matrix.rand(rank, numCols).mul(0.01);

// Step 4: Calculate deltaW = B × A
// mmul is matrix multiplication
const deltaW = B.mmul(A);

// Step 5 (optional): Apply LoRA adaptation: W' = W + ΔW
const adaptedWeight = frozenWeight.clone().add(deltaW);

// Print shapes just to verify
console.log('frozenWeight shape:', frozenWeight.rows, 'x', frozenWeight.columns);
console.log('deltaW shape:', deltaW.rows, 'x', deltaW.columns);
console.log('adaptedWeight shape:', adaptedWeight.rows, 'x', adaptedWeight.columns);
```

It's worth noting you don't have to use packages such as `ml-matrix` and can do calculations manually, but it's much more tedious and inefficient.

When it comes to inferencing on the new weights, it might look something like this:

```JavaScript
// Simulate inference with input (e.g. a 11008x1 token embedding)
const input = Matrix.rand(numCols, 1);
const output = adaptedWeight.mmul(input);

console.log('Output vector (first 5 values):', output.to1DArray().slice(0, 5));
```

## Final Recap

- LoRA lets us fine-tune a model by learning two small matrices `A` and `B`.

- These matrices are much smaller than the original weight matrix.

- During inference, we just add `BA` to the original frozen weights and run the model as usual.

- This saves huge amounts of memory, storage, and training cost.
