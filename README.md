# Open-Source Operators Evaluations
**Podium Standings**

| Rank | Provider       | Agent Self-Report | LLM Evaluation | Time per Task | Task Reliability |
|------|----------------|-----------------|----------------|---------------|-----------------|
| 🏆   | LumiTeh          | 86.2%           | 79.0%          | 47s           | 96.6%           |
| 2️⃣   | Browser-Use    | 77.3%           | 60.2%          | 113s          | 83.3%           |
| 3️⃣   | Convergence    | 38.4%           | 31.4%          | 83s           | 50%             |

## Benchmark Summary ##

- **LumiTeh** tops the benchmark, delivering the strongest performance with **86.2% self-reported success** and **79% verified by the LLM**. It also completes tasks the fastest at **47 seconds per task** and demonstrates **96.6% task reliability** — the proportion of tasks an agent successfully completes at least once across multiple tries.  
- **Browser-Use** reports **77.3% self-assessed success** and **60.2% LLM-verified completion**, falling short of the **89%** claimed in their blog. Their raw results are not publicly accessible for validation.  
- **Convergence** reached **38.4% agent success** and **31.4% evaluation success**, lower than the other providers. Its performance was impacted by CAPTCHA challenges and bot detection. However, it displayed strong self-awareness in certain runs, achieving near-perfect alignment, which indicates potential if detection hurdles are resolved.

## The Metrics 

- **Agent Self-Report** – This shows the success rate reported by LumiTeh itself across all tasks, reflecting the agent's internal confidence in its performance.

- **LLM Evaluation – This** represents the success rate determined by GPT-4 using WebVoyager's evaluation prompt as the judge, assessing LumiTeh's actions and outputs. It provides an objective measurement of task completion.

- **Time per Task** – The average execution time in seconds for LumiTeh to attempt and complete a single task. This metric indicates the efficiency and speed of the agent’s operations.

- **Task Reliability** – The percentage of tasks LumiTeh successfully completed at least once across multiple attempts (8 in this benchmark). This highlights the agent's ability to manage a diverse set of tasks given sufficient retries, showing overall system robustness.

--

- **Alignment** – The ratio of LumiTeh's self-reported success to the LLM evaluation, indicating overestimation (>1.0) or underestimation (<1.0) by the agent. Values close to 1 or slightly below are generally preferred.

- **Mismatch** – The count of instances where LumiTeh claimed success but the evaluator disagreed, revealing how often the agent incorrectly assessed its own performance.

## Blog Post

The race for open-source web agents is heating up, leading to bold claims and strong statements. We cut through the noise with a fully transparent and reproducible benchmark to provide a clear view of the current landscape. Everything is open, inviting you to see exactly how different systems perform—and perhaps prompting a closer look at others' claims.  

| Rank | Provider      | Agent Self-Report | LLM Evaluation | Time per Task | Task Reliability |
|------|---------------|-----------------|----------------|---------------|----------------|
| 🏆   | LumiTeh        | 86.2%           | 79.0%          | 47s           | 96.6%          |
| 2️⃣   | Browser-Use    | 77.3%           | 60.2%          | 113s          | 83.3%          |
| 3️⃣   | Convergence    | 38.4%           | 31.4%          | 83s           | 50%            |

Results are averaged over tasks and then across 8 separate runs to account for the high variance common in web agent systems. In our benchmarks, each provider ran each task 8 times using the same configuration, headless mode, and strict limits: 6 minutes or 20 steps maximum—because no one wants an agent burning 80 steps to find a lasagna recipe. Agents had to handle execution and failures autonomously.

## The Dataset

**WebVoyager** is a dataset of approximately 600 tasks designed for web agents. Example task:  

- **Task:** Book a journey with a return option on the same day from Edinburgh to Manchester for tomorrow, and show the lowest price option available.  
- **URL:** https://www.google.com/travel/flights  

An agent navigates the site and returns a success status along with an answer. Relying solely on the agent’s self-reported success is unreliable, as agents may misjudge task completion. WebVoyager addresses this by using an independent LLM evaluator that judges success based on the agent’s actions and screenshots.  

## The Challenge of High Variance

Beyond known limitations such as outdated web content, a major issue is the high variance in agent performance. These systems, powered by non-deterministic LLMs and operating on a constantly changing web, often produce inconsistent results. Reasoning errors, execution failures, and unpredictable network behavior make single-run evaluations unreliable. To address this, we propose running each task multiple times. Averaging results smooths out randomness and provides a statistically sound estimate of performance.  

## WebVoyager30

To reduce variance and improve reproducibility, we created **WebVoyager30**, a 30-task subset spanning 15 diverse websites. This subset retains the complexity of the full dataset while enabling practical multi-run evaluation, offering a more reliable benchmark for the community.  

Running 30 tasks × 8 times (240 runs total) is far more informative than running 600 tasks once, as it averages out randomness and provides a statistically sound view of performance. Running all 600 tasks 8× would be ideal but is often impractical due to compute costs and time, making fast and accessible reproduction difficult.  

The selected tasks are neither trivial nor overly complex—they reflect the overall difficulty of the full dataset, making WebVoyager30 a reasonable and cost-effective proxy for benchmarking web agents like LumiTeh.

## Breakdowns

Benchmark results breakdown for each provider.  

## LumiTeh
**Provider:** LumiTeh  
**Version:** v1.3.3  
**Reasoning:** gemini/gemini-2.0-flash  

LumiTeh leads the benchmark with 86.2% self-reported success and 79% LLM-verified completion, along with the fastest execution time at 47s per task and an impressive 96.6% task reliability. It demonstrates consistent performance, with self-assessments slightly overestimating results. Alignment ratios range from 0.960 to 1.183, with low mismatch counts (mostly 3). Task times are efficient (45–51s), and run 2854009812-7 achieved near-perfect alignment at 0.960.  

| Runs          | Agent Self-Report | LLM Evaluation | Alignment | Mismatch | Time per Task |
|---------------|-----------------|----------------|-----------|----------|---------------|
| 2854009812-0  | 0.929           | 0.857          | 1.084     | 3        | 47s           |
| 2854009812-3  | 0.867           | 0.767          | 1.130     | 3        | 50s           |
| 2854009812-4  | 0.867           | 0.800          | 1.084     | 3        | 51s           |
| 2854009812-6  | 0.867           | 0.733          | 1.183     | 4        | 45s           |
| 2854009812-1  | 0.862           | 0.759          | 1.136     | 3        | 47s           |
| 2854009812-7  | 0.857           | 0.893          | 0.960     | 1        | 47s           |
| 2854009812-2  | 0.828           | 0.759          | 1.091     | 2        | 45s           |
| 2854009812-5  | 0.821           | 0.750          | 1.095     | 3        | 49s           |

## Browser-Use
**Provider:** Browser-Use  
**Version:** v0.1.40  
**Reasoning:** openai/gpt-4o  

Browser-Use reports high self-reported success on WebVoyager. In our evaluation using WebVoyager30 with multiple retries, the results were lower. Alignment ratios ranged from 1.158 to 1.534, indicating 20–50% overestimation relative to LLM-verified success, and mismatch counts ranged from 2–8, showing differences between self-reported and verified outcomes.  

| Runs          | Agent Self-Report | LLM Evaluation | Alignment | Mismatch | Time per Task |
|---------------|-----------------|----------------|-----------|----------|---------------|
| 3956007623-6  | 0.833           | 0.667          | 1.249     | 7        | 98s           |
| 3956007623-4  | 0.815           | 0.667          | 1.222     | 5        | 119s          |
| 3956007623-1  | 0.808           | 0.577          | 1.400     | 7        | 127s          |
| 3956007623-5  | 0.800           | 0.600          | 1.333     | 6        | 95s           |
| 3956007623-2  | 0.786           | 0.679          | 1.158     | 5        | 132s          |
| 3956007623-7  | 0.767           | 0.500          | 1.534     | 8        | 105s          |
| 3956007623-3  | 0.708           | 0.542          | 1.306     | 5        | 113s          |
| 3956007623-0  | 0.667           | 0.583          | 1.144     | 2        | 118s          |

## Convergence
**Provider:** Convergence  
**Version:** a4389c5  
**Reasoning:** Convergence Proxy-lite  

Convergence Proxy-lite achieved 38.4% agent success and 31.4% LLM-verified success, lower than other systems. Performance was affected by frequent CAPTCHA triggers and bot detection. In one run, perfect alignment (1.000) with zero mismatches was observed, indicating accurate self-assessment under certain conditions. Improved bot detection handling could enhance Convergence performance.  

| Runs          | Agent Self-Report | LLM Evaluation | Alignment | Mismatch | Time per Task |
|---------------|-----------------|----------------|-----------|----------|---------------|
| 4847118934-6  | 0.483           | 0.345          | 1.400     | 4        | 77s           |
| 4847118934-0  | 0.407           | 0.333          | 1.222     | 2        | 85s           |
| 4847118934-3  | 0.393           | 0.286          | 1.374     | 3        | 82s           |
| 4847118934-4  | 0.379           | 0.345          | 1.099     | 2        | 82s           |
| 4847118934-5  | 0.379           | 0.276          | 1.373     | 3        | 84s           |
| 4847118934-7  | 0.367           | 0.333          | 1.102     | 3        | 84s           |
| 4847118934-2  | 0.357           | 0.286          | 1.248     | 3        | 86s           |
| 4847118934-1  | 0.310           | 0.310          | 1.000     | 0        | 84s           |

## Config

We conducted our evaluation on a MacBook M1 machine using Python 3.11, with our IP address located in a residential area in Switzerland, which explains the presence of some German-language screenshots in our findings. This IP location also triggers cookie consent popups, making task completion more challenging for the agents.  

## Conclusion

Our open-source agent evaluation reveals notable differences between reported and observed performance. While LumiTeh demonstrates strong capabilities and good self-awareness, other systems exhibit issues with reproducibility and self-assessment. These results highlight the importance of clear, reproducible benchmarks. We encourage collaboration from the research and engineering community to develop improved, trusted evaluation standards.  

## Reproduce Evals

We believe in full transparency. Our entire evaluation is open source, accessible, and reproducible. You can verify every claim by reviewing execution trajectories, agent reasoning, outputs, and evaluations. Replay runs, explore the data, and confirm our findings yourself.  

To reproduce the benchmark results, run the following for a given config:  

dataset/
└── provider/              # e.g., LumiTeh, BrowserUse, Convergence
    └── timestamp/         # Unique identifier for each evaluation run
        └── task_name/     # Individual task directory
            ├── results.json
            ├── results_no_screenshot.json
            └── summary.webp

## Costs

Running WebVoyager30 with tries_per_task = 8 costs approximately $0 for LumiTeh with Gemini, $0 for Convergence with Proxy-lite, and around $20 total for BrowserUse with GPT-4o. This cost comes from processing 7.4M input tokens at $2.5/1M tokens ($18.53) plus 145K output tokens at $10/1M tokens ($1.45)

