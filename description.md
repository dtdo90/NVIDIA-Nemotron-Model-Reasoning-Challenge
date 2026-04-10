## Overview
Develop techniques that improve reasoning accuracy using NVIDIA Nemotron models.

Participants will experiment with prompting, data pipelines, and lightweight fine-tuning while evaluating their approaches on a new reasoning benchmark developed by NVIDIA Research.

## Description
Reasoning benchmarks are a useful way to measure progress on structured tasks. When approaches and results are shared openly, the community can compare methods, reproduce improvements, and iterate more effectively.

Today, reasoning improvements are explored across many independent efforts - often using different datasets, prompts, and evaluation setups - making direct comparison difficult. A shared benchmark and common baseline model allow techniques to be tested and compared more consistently.

While language models perform strongly on many tasks, structured reasoning benchmarks remain an active area of research and optimization.

In this competition, participants will work from a shared Nemotron 3 Nano baseline and a novel reasoning benchmark developed by NVIDIA Research. Nemotron provides an open foundation for this challenge, including openly available models, datasets, and training recipes that participants can build on or adapt within their own workflows.

You may experiment with:

* Prompting strategies
* Data filtering and curation
* Synthetic data generation
* Reinforcement learning
* Lightweight fine-tuning
* Or other approaches of your choice

Participants may use any training framework, tooling, or workflow to produce their LoRA adapter. NVIDIA-provided recipes are optional starting points - competitors are free to use other ecosystems and libraries (e.g., Hugging Face, Unsloth, Axolotl, TRL, or similar tooling).

The only requirement is that the final submission produces a compatible LoRA adapter for the Nemotron-3-Nano-30B base model.

Multiple valid solution paths are expected. Clear documentation - including notebooks and write-ups - is encouraged (and required for prize eligibility) to support reproducibility and communal learning.

By bringing models, datasets, and evaluation into an open, shared environment, this challenge creates an opportunity for collaborative iteration - strengthening open reasoning workflows that others can study, reuse, and extend.

## Evaluation

Submissions are evaluated based on their Accuracy in solving the provided tasks. The NVIDIA Nemotron-3-Nano-30B model is loaded with your LoRA adapter (which must include an adapter_config.json) using the vLLM inference engine. For each test case, the model is prompted to generate a response and instructed to place its final answer within a \boxed{} LaTeX command. The metric extracts the final answer from the generated text, prioritizing content within the boxed format while falling back to other heuristic patterns or the last numeric value found. A prediction is graded as correct if it matches the ground truth either exactly as a string or within a relative numerical tolerance of $10^{-2}$. The final score is the proportion of correctly answered questions.

You may view the implementation of the metric here: NVIDIA Nemotron Metric. It is being run with the following parameters:

Parameters and Value
* max_lora_rank: 32
* max_tokens: 7680
* top_p: 1.0
* temperature: 0.0
* max_num_seqs: 64
* gpu_memory_utilization: 0.85
* max_model_len: 8192


## Dataset Description
This dataset comprises a collection of logical reasoning puzzles requiring the identification and application of underlying transformation rules. The puzzles cover various domains, such as bit manipulation and algebraic equations.

### File and Field Information
**train.csv:** The training set containing puzzles and their corresponding solutions.
* id - A unique identifier for each puzzle.
* prompt - The puzzle description, including input-output examples and the specific 
* instance to be solved.
* answer - The ground truth solution for the puzzle.

**test.csv:** A sample test set to help you author your submissions. When your submission is scored, this will be replaced by a test set of several hundred problems.
* id - A unique identifier for each puzzle.
* prompt - As in train.csv.

Note that your submission must be a file submission.zip containing a LoRA adapter. See the Evaluation page for details.


## Submitting
You must submit a LoRA adapter of rank at most 32 for the NVIDIA Nemotron-3-Nano-30B model packaged into a submission.zip file. You may consider adapting the NVIDIA Nemotron Submission Demo to produce your submission.