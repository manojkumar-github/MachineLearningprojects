I would structure this presentation around representative AI workloads you have actually worked on, how they execute across CPUs and GPUs, how you benchmark their performance, and what those workloads mean for AMD’s hardware and software optimization strategy.

Given your background at Yahoo and your earlier enterprise AI/ML roles, you have a strong story to tell. You have experience spanning traditional ML, production LLM inference, RAG, agentic workflows, and inference benchmarking. That breadth is more valuable to your new team than a presentation focused exclusively on agentic AI.

Your audience consists primarily of hardware benchmarking engineers. They will likely be most interested in understanding:

* What real applications run on the hardware.
* How workloads behave in production.
* Which components execute on CPUs versus GPUs.
* What creates performance bottlenecks.
* Which benchmarks and metrics represent real-world performance.
* How future AI workloads might change their benchmarking priorities.

Here is how I would approach the presentation.

Recommended presentation

From Production AI Workloads to Hardware Performance: Lessons from Yahoo and Beyond

Target duration: 25–30 minutes, followed by discussion.

The narrative should follow this progression:

My experience → Representative AI workloads → CPU/GPU execution → Performance characteristics → Benchmarking methodology → Future opportunities for AMD

⸻

Part 1. Establish the context

Slide 1 — My journey through AI workloads

Purpose: Introduce your background and establish why your experience is relevant to the team.

Keep this to one slide.

Your career progression provides a natural narrative:

Experience	Workload perspective
Pearson	Production NLP and deep-learning applications
Metrix engagements	Enterprise ML platforms, distributed ML pipelines, and RAG-based systems
Yahoo	Production LLM inference, inference optimization, benchmarking, and agentic AI workflows
AMD	Applying workload understanding to hardware performance and benchmarking

These categories are based on your resume. Principal-AI-Inference-Performance-Benchmarking-Engineer.pdf

Don’t spend several minutes walking through your résumé. The goal is to establish the progression from traditional ML to modern production AI.

Key message: My experience spans both the applications that generate AI workloads and the infrastructure that executes them.

Slide 2 — How AI workloads have evolved

Introduce four workload categories.

Workload	Representative example	Primary performance concerns
Traditional ML inference	Classification, prediction, NLP	Throughput, latency, CPU/GPU utilization
LLM inference	Text generation and summarization	Time to first token, token generation speed, memory capacity
Retrieval-augmented generation	Retrieve documents, then generate an answer	Retrieval latency, prompt processing, inference latency
Agentic AI	Multi-step workflows using models and tools	End-to-end latency, repeated inference, orchestration, task success

This is where you introduce an important distinction.

Traditional ML workloads often have a relatively well-defined execution path. LLM and agentic applications can have more variable execution patterns, particularly when context lengths, generated output lengths, or the number of inference calls vary.

Key message: The workload is no longer defined only by the model. It is increasingly defined by how the application uses the model.

⸻

Part 2. Walk through representative workloads from your experience

This should be the heart of the presentation.

I recommend choosing three representative workloads rather than trying to cover every project you’ve worked on.

My suggested choices are:

1. Production LLM inference at Yahoo.
2. RAG and enterprise AI applications.
3. Agentic AI workflows.

These are supported by your résumé. However, the precise execution characteristics and performance findings need to come from your actual experience.

Slide 3 — Workload 1: Production LLM inference at Yahoo

Your resume describes production deployment of fine-tuned Qwen variants, Flan-T5, and Llama models on GCP, alongside inference optimization and benchmarking.

It also describes an LLM platform supporting 220M+ users and billions of requests per day, with traffic scaling from approximately 10K to 50K QPS. Separately, your Yahoo Mail Intelligence experience references more than one billion daily requests. Be clear about which traffic figure applies to which platform or service when presenting them. Principal-AI-Inference-Performance-Benchmarking-Engineer.pdf

What to present

A. The workload

Explain what an inference request looks like:

* A request arrives from an application.
* The system prepares the model input.
* The serving runtime schedules inference.
* The model generates a prediction or text response.
* The output is processed and returned.

B. The infrastructure

Show the major components:

Application / Client
        |
        v
API / LLM Gateway
        |
        v
Request Routing and Scheduling
        |
        v
CPU Host
  - Request processing
  - Tokenization / preprocessing
  - Runtime coordination
        |
        v
GPU Inference
  - Model execution
  - Prefill
  - Token generation
        |
        v
Postprocessing / Response
        |
        v
Application / Client

This is a conceptual serving diagram, not a claim that every component in your Yahoo deployment followed precisely this sequence or ran on a particular processor.

C. The performance questions

Explain what you need to measure in a production inference environment:

* Time to first token (TTFT).
* Inter-token latency.
* End-to-end request latency.
* Throughput at different concurrency levels.
* GPU utilization and memory consumption.
* Error rates and overload behavior.

D. What you personally optimized

Your resume identifies quantization methods including AWQ, GPTQ, and GGUF; FlashAttention; dynamic batching; and serving technologies such as vLLM, TensorRT, ONNX, PyTorch, and Triton.

Discuss two or three of these techniques in enough detail to explain why they mattered for your actual workload.

What the AMD team should learn: A production inference workload is a complete serving pipeline, not simply a model running on a GPU.

⸻

Slide 4 — Workload 1: What determines LLM inference performance?

Now move from the architecture to the behavior of the workload.

Use two phases to explain model execution.

Phase A: Prefill

The model processes the input prompt and builds the state needed for generation.

Important characteristics include:

* Input sequence length.
* Batch size and concurrent requests.
* Compute throughput.
* Memory traffic and attention implementation.

Phase B: Decode

The model generates output tokens iteratively.

Important characteristics include:

* Output sequence length.
* Memory bandwidth and KV-cache behavior.
* Number of concurrent sequences.
* Scheduling and batching policy.

Show this simple flow:

Input Prompt
     |
     v
   PREFILL
 Process Input
     |
     v
 KV Cache / Generation State
     |
     v
    DECODE
 Generate Tokens Iteratively
     |
     v
 Final Response

Then explain why two benchmarks using the same model can produce different results when input length, output length, concurrency, or batching changes.

Question to raise with the team: Are we measuring the performance characteristics that dominate the target production workload, or merely one convenient model configuration?

That question connects your inference experience directly to hardware benchmarking.

⸻

Slide 5 — Workload 2: Retrieval-augmented generation

Your résumé includes production RAG systems and enterprise GenAI platforms from your Metrix engagements. Principal-AI-Inference-Performance-Benchmarking-Engineer.pdf

RAG is useful to include because it demonstrates how AI applications combine GPU inference with other components that may execute on CPUs or separate services.

Show the complete request path

User Question
      |
      v
Application / Orchestrator
      |
      v
Query Processing
      |
      v
Retrieval System
      |
      v
Relevant Documents
      |
      v
Prompt Construction
      |
      v
LLM Inference on GPU
      |
      v
Response Processing
      |
      v
Final Answer

The CPU commonly handles orchestration, query processing, and prompt construction. Retrieval may execute on a CPU-based service, a specialized vector-search engine, or another accelerated system. The LLM typically runs on a GPU when deployed for GPU inference.

What makes RAG different from standalone inference?

The total response time includes retrieval and preprocessing as well as model execution.

For example, if retrieval takes longer than expected, a faster GPU might improve generation performance without proportionately improving the user’s total response time.

That is a general architectural observation, not a claim about the measurements in your own projects.

What to discuss

* How retrieval changes the input context.
* How context length affects prefill.
* Whether retrieval or inference dominates latency.
* How CPU processing, storage, and network delays affect the overall pipeline.
* How to benchmark the complete request rather than only the generation stage.

Key message: Application-level performance depends on the interaction between retrieval, preprocessing, and inference.

⸻

Slide 6 — Workload 3: Agentic AI workflows

Your résumé describes multi-agent and tool-augmented LLM systems, multi-step reasoning, orchestration, context and memory frameworks, and MCP-based integrations. Principal-AI-Inference-Performance-Benchmarking-Engineer.pdf

This is where your recent agentic AI engineering experience becomes particularly relevant.

Explain the execution loop

           User Request
                 |
                 v
      CPU: Agent Orchestration
                 |
                 v
       GPU: Model Inference
                 |
                 v
       CPU: Interpret Response
                 |
          +------+------+
          |             |
       Tool Call     Final Answer
          |
          v
 CPU / External Tool Execution
          |
          v
      Tool Result
          |
          v
  CPU: Update Agent Context
          |
          +------> Next Model Call

This is a simplified representation. Some implementations distribute orchestration across multiple services, and tool execution may use CPUs, specialized accelerators, or remote systems.

Explain the CPU/GPU relationship

CPU responsibilities commonly include:

* Agent state and workflow orchestration.
* Tool selection and invocation.
* Context construction.
* Request scheduling and coordination.
* Processing tool responses.
* Deciding whether another model call is necessary.

GPU responsibilities commonly include:

* Transformer model execution.
* Prompt prefill.
* Token generation.
* Other supported tensor operations.

The CPU is not necessarily busy during every GPU operation, and the GPU may be serving multiple requests concurrently. The key is understanding the dependencies and overlap between stages.

What makes agentic workloads interesting?

A single user task may trigger several inference calls, tool executions, and context updates.

Consequently, the total task latency depends on more than the latency of one inference call.

This creates new benchmarking questions:

* How many inference calls are required per task?
* How much time is spent executing tools?
* How much CPU/GPU overlap is possible?
* How does concurrency affect the execution loop?
* How many tasks finish within a target latency?

Key message: Agentic workloads shift the performance question from individual model requests toward complete multi-step execution.

⸻

Part 3. Connect the workloads to hardware performance

Slide 7 — Three workloads, three performance profiles

Bring the previous examples together.

Characteristic	LLM inference	RAG	Agentic AI
Main execution pattern	Prefill and decode	Retrieval followed by inference	Repeated inference and conditional execution
CPU role	Serving coordination and preprocessing	Retrieval coordination, preprocessing, and orchestration	Workflow control, tools, state, and orchestration
GPU role	Model inference	Model inference	Model inference across one or more steps
Major variables	Context length, output length, concurrency	Retrieval time, context length, inference performance	Number of model calls, tool latency, branching, concurrency
Useful end-to-end metric	Request latency and throughput	Complete query latency	Task latency and successful task throughput

Use this slide to reinforce that different workloads can exercise the same hardware in very different ways.

Slide 8 — Where does the time go?

This is a particularly effective slide for a benchmarking audience.

Use a hypothetical ten-second agentic request:

Stage	Illustrative duration
Model inference across multiple calls	4.0 seconds
Tool and external-service execution	3.0 seconds
Orchestration and preprocessing	1.5 seconds
Other processing and coordination	1.5 seconds
Total	10.0 seconds

Label the values clearly as illustrative, not measurements from Yahoo.

Now suppose the inference component becomes twice as fast. Its duration drops from four seconds to two seconds, and total task latency falls from ten seconds to eight seconds.

That is a 20% reduction in total latency.

The example illustrates why component-level benchmarking and end-to-end benchmarking are complementary.

It also creates a practical question: if AMD improves GPU inference performance, how much of that improvement translates into the application the customer runs?

The answer depends on the workload’s execution profile.

Slide 9 — What I measured and optimized in production

Use your actual experience here rather than adding more generic architecture diagrams.

Your résumé describes:

* Benchmarking and profiling across A100, H100, G2-L4, and G4 NVIDIA RTX PRO 6000 configurations.
* TTFT, tokens per second, latency, memory footprint, and GPU utilization measurements.
* Monitoring of errors, including HTTP 429 responses.
* GPU fleet sizing and client-side concurrency planning.
* Reported improvements of 45%+ in latency and 50%+ in GPU cost. Principal-AI-Inference-Performance-Benchmarking-Engineer.pdf

Organize these into three categories.

Workload characterization

* Input and output characteristics.
* Request rate and concurrency.
* Model selection and serving configuration.

Performance measurement

* Latency and throughput.
* Memory utilization and GPU utilization.
* Error rates and system behavior under load.

Optimization

* Model and runtime configuration.
* Quantization and attention optimization.
* Batching and concurrency.
* Capacity planning.

Only attribute a specific improvement to a specific optimization if your own measurements support that causal relationship.

⸻

Part 4. How should AMD benchmark these workloads?

Slide 10 — A layered benchmarking strategy

I would propose three levels.

Level	Benchmark focus	Primary question
Hardware	Compute, memory bandwidth, capacity, interconnect	What can the hardware deliver?
Model serving	TTFT, inter-token latency, throughput, memory	How efficiently does the system execute inference?
Application	End-to-end latency, task throughput, success rate, cost	How well does the complete workload perform?

Explain that the levels should complement one another.

Hardware benchmarks help isolate capabilities and bottlenecks. Serving benchmarks characterize model execution. Application benchmarks evaluate whether those improvements translate into better outcomes.

Slide 11 — How I would design a representative workload suite

Propose a suite with four workload families:

1. Interactive LLM inference: Short and medium prompts, variable output lengths, and concurrent requests.
2. Long-context inference: Longer inputs and their effects on prefill, memory consumption, and generation.
3. RAG: Retrieval, context assembly, inference, and response processing.
4. Agentic workflows: Repeated inference, tool use, orchestration, and conditional execution.

For each workload, define:

* Model and runtime.
* Input and output distributions.
* Concurrency and request arrival pattern.
* Latency requirements.
* Hardware and software configuration.
* Metrics and measurement methodology.

The suite should use representative traces or distributions wherever possible, rather than relying exclusively on one fixed input and output length.

This is a proposed methodology for AMD to explore, not a claim that these workload families are absent from existing AMD benchmarks.

Slide 12 — Metrics that connect hardware performance to application value

Focus on the metrics that matter across workload layers.

Metric	Why it matters
TTFT	Measures responsiveness for streaming inference
Inter-token latency	Measures generation cadence
Throughput at a latency target	Captures useful serving capacity
GPU utilization	Helps identify underutilization and saturation
Memory footprint	Helps determine model fit and concurrency limits
End-to-end latency	Captures the complete application experience
Successful tasks per second	Measures completed work for agentic applications
Cost per completed task	Connects performance to operational efficiency

One caveat: task success must be defined consistently, especially when comparing different models or agent configurations.

⸻

Part 5. The future and the opportunity for AMD

Slide 13 — The future of AI workloads

Organize this into three trends.

1. More multi-step workflows

Applications increasingly combine inference with retrieval, tools, and orchestration.

2. More heterogeneous execution

CPUs, GPUs, memory, networking, and external services all contribute to the complete application.

3. More emphasis on useful work

Performance evaluation increasingly needs to account for latency, concurrency, quality, reliability, and cost alongside raw throughput.

For AMD, the implication is to continue connecting hardware and model-serving measurements with representative application workloads.

Slide 14 — Opportunities for AMD workload benchmarking

Offer three concrete directions for discussion:

* Workload realism: Build repeatable benchmarks that reflect representative production request patterns.
* Cross-layer profiling: Correlate CPU execution, GPU activity, memory behavior, and end-to-end latency.
* Application-level performance: Measure successful workload completion and performance under defined service constraints.

This slide should invite collaboration, not imply that AMD lacks these capabilities.

Your experience gives you a useful perspective on how to define the workloads, interpret the measurements, and identify gaps between isolated benchmark results and application performance.

Slide 15 — Key takeaways

Close with three statements:

1. Workload characteristics determine what hardware performance matters.
2. CPU orchestration and GPU inference must be understood as parts of one execution system.
3. Representative benchmarking should connect hardware capabilities to real application outcomes.

End with a discussion question:

How can we build a benchmark suite that reflects the AI workloads customers will actually run on AMD hardware?

⸻

How I would deliver this presentation

Although the structure contains 15 slides, I would spend most of the time on slides 3–9. Those slides demonstrate your actual technical experience and give the hardware team something they can apply to their work.

The industry overview should be brief. The future section should lead to a concrete benchmarking discussion rather than become a speculative survey of AI trends.

One final recommendation: use actual workload data wherever you can, and clearly label hypothetical examples. A real inference trace, latency breakdown, concurrency curve, or before-and-after optimization result will be much more compelling to this audience than another conceptual diagram.

Your strongest contribution is not predicting every future AI trend. It is showing how to reason from application behavior to measurable hardware performance.
