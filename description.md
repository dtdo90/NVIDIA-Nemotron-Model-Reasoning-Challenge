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

## Timeline
* March 16, 2026 - Start Date.
* April 9, 2026 - Midpoint Cut-off Date
* June 8, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.
* June 8, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.
* June 15, 2026 - Competition End Date & Final Submission Deadline.
* June 22, 2026 - Open Contribution Award Submissions Deadline
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.


## Prizes
To be eligible for any prize, teams must publish a public Kaggle notebook and solution write-up documenting the methods, datasets, and techniques used to produce the submission. Submissions without qualifying public documentation may be deemed ineligible for prizes.

### Final Leaderboard Prizes

* 1st Place - $25,000 + 5 DGX Sparks
* 2nd Place - $15,000 + 2 DGX Sparks
* 3rd Place - $5,000 + 1 DGX Sparks

Note: A total of eight (8) NVIDIA DGX Spark systems (Approximate Retail Value: $4,699 per system) will be awarded based on final leaderboard placement. If any team has fewer eligible members than the number of DGX Spark systems allocated for that placement, or if any team member is ineligible to receive hardware due to export, shipping, or regional restrictions, any remaining units will cascade to the next highest-ranked teams on the final leaderboard until all eight (8) DGX Spark systems have been awarded.

Each eligible participant may receive no more than one (1) DGX Spark, and only officially registered team members are eligible to receive hardware prizes. NVIDIA reserves the right to verify team membership and eligibility prior to awarding hardware prizes.

### Open Progress Prize (Mid-Competition Milestone)
Open Progress Prize: $5,000 + 1 DGX Sparks

* Awarded to the team with the highest leaderboard score as of the Midpoint Cut-off Date: April 9, 2026.
* Methodology submissions Cut-off Date: April 16, 2026.
* Winners will be announced during Cloud NEXT (April 22-24, 2026)

If the top-ranked submission at the cutoff date does not meet these requirements, the prize will be awarded to the next highest eligible submission.

In the event of a tie, the prize will be awarded to the team whose qualifying submission was submitted earliest.

Each eligible participant may receive no more than one (1) DGX Spark.

### Open Contribution Awards
The Open Contribution Awards recognize techniques that meaningfully advance reasoning performance using Nemotron models.

Three awards will be granted:

* Best Data/Synthetic Data Method - 1 DGX Spark
* Best RL Method - 1 DGX Spark
* Best Fine-tuning Method - 1 DGX Spark

Participants must submit their entry for these awards through this form linking to their notebook and clearly identifying the category being entered.

Only submissions ranking within the top 10% of the final leaderboard will be considered for Open Contribution Awards.

## Citation
Jamil C Semaan, Jean-Francois Puget, Christof Henkel, Yi Dong, Addison Howard, Ashley Oldacre, Ryan Holbrook, Chris Alexiuk, and Rebecca Kao. NVIDIA Nemotron Model Reasoning Challenge. https://kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge, 2026. Kaggle.