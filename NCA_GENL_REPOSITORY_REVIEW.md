# NCA-GENL Repository Review & Correction Plan

> Repository: https://github.com/nxthanh2904/nca-genl  
> Review date: 2026-11-03  
> Scope: `README.md`, 11 knowledge modules (`Module_01`–`Module_11`), `NCA_GENL_Exam_Guide_and_Tips.md`, `index.html`  
> Purpose: Improve factual accuracy and alignment with the NVIDIA NCA-GENL exam blueprint, with special attention to software developers.  
> Status: **Applied and verified across all source files (2026-11-03)**.

## 0. Executive summary

The repository covers many relevant topics: AI/ML fundamentals, transformers, model choice, prompt engineering, RAG, PEFT, training, NVIDIA tooling, and trustworthy AI. It is useful as a study guide, but **not every assertion should be memorized verbatim**. Several statements are excessively absolute; some blur useful distinctions; and Experiment Design / Data Analysis & Visualization are less developed than their examination weight warrants.

**Official NVIDIA exam information, checked 2026-11-03:**

- NCA-GENL is an Associate-level certification.
- Exam duration: **60 minutes**; language: **English**.
- The certification page says **50 questions** in its overview and **50–60 multiple-choice questions** in its detailed exam information. Because the official page itself contains both, do **not** present either as an unqualified fixed count; note this discrepancy and check the booking information closer to the exam.
- Official domain weights: **Core ML & AI Knowledge 30%**, **Software Development 24%**, **Experimentation 22%**, **Data Analysis & Visualization 14%**, **Trustworthy AI 10%**.
- Official emphasis includes **experiment design**, **data analysis/visualization**, **Python libraries for LLMs**, and **LLM integration/deployment**.

Official source: https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/

### Priority definitions

- **P0 — Incorrect / potentially misleading:** fix before using as definitive study material.
- **P1 — Overgeneralized / requires caveats:** clarify to prevent mistakes in scenario questions.
- **P2 — Improvements:** completeness, examples, navigation, exam usefulness.

## 1. Verified corrections: exact locations and replacement wording

The snippets below are short identifying phrases from the current repository, not necessarily complete original paragraphs. Apply replacements to the corresponding section and check related diagrams, tables, quizzes and README summaries.

### FIX-01 — BF16 does not eliminate overflow [P0]

- **File:** `Module_01_AI_Infrastructure.md` §3.4.
- **Find:** `không bao giờ sợ tràn số (Overflow)`.
- **Issue:** BF16 has 8 exponent bits like FP32, offering a **similar dynamic range** but much less significand precision. Overflow, underflow and numerical instability can still occur.
- **Replace with:**

  > `BF16` dùng 1 bit dấu, 8 bit số mũ và 7 bit phần trị. Nhờ số bit mũ bằng FP32, BF16 giữ được dải động rộng hơn FP16, thường giảm nhu cầu loss scaling trong mixed-precision training. Tuy nhiên BF16 **không loại bỏ hoàn toàn overflow/underflow** và có độ chính xác phần trị thấp hơn FP16.

- **Exam cue:** FP16 = more mantissa precision; BF16 = wider dynamic range.

### FIX-02 — Pinned memory is not mandatory for all CPU–GPU copies [P0]

- **File:** `Module_01_AI_Infrastructure.md` §3.5.
- **Find:** `bắt buộc phải cấp phát ... Pinned Memory`.
- **Issue:** CUDA supports transfers involving pageable host memory, potentially using staging; page-locked/pinned memory is particularly important for effective asynchronous transfers and throughput.
- **Replace with:**

  > CUDA có thể truyền dữ liệu giữa host và device từ cả pageable memory lẫn page-locked (pinned) memory. Pinned memory giúp tránh một số bước staging và thường cần để `cudaMemcpyAsync` có hành vi bất đồng bộ đúng mong đợi, đồng thời cải thiện thông lượng. Không phải **mọi** host-to-device copy đều bắt buộc tự cấp phát pinned memory.

- **Reference:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

### FIX-03 — GenAI is not a strict subset of DL in every definition [P1]

- **File:** `Module_02_AIML_Fundamentals.md` §2.1.
- **Find:** `Generative AI là tập con của Deep Learning`.
- **Replace with:**

  > AI bao gồm ML, và Deep Learning là một nhánh của ML. **Generative AI mô tả loại nhiệm vụ/hệ thống tạo dữ liệu mới**, trong đó các hệ thống hiện đại chủ yếu dựa trên Deep Learning (LLMs, diffusion models). Không nên coi toàn bộ mô hình sinh đều bắt buộc là Deep Learning: một số mô hình xác suất truyền thống cũng có tính sinh.

- **Action:** revise the nested-circle diagram to show this nuance rather than an unconditional mathematical containment.

### FIX-04 — Transformer is not “100% parallel” [P0]

- **File:** `Module_03_Generative_AI_and_LLM.md` §4.4, related Mermaid diagram.
- **Find:** `Song song 100% trên GPU` and `Tận dụng 100% Tensor Cores`.
- **Replace with:**

  > Transformer có khả năng song song hóa tốt trên các vị trí token **trong quá trình huấn luyện** so với RNN truyền thống. Tuy nhiên, với mô hình sinh văn bản autoregressive, **quá trình decoding thường tạo token tuần tự**, dù các phép toán bên trong mỗi bước có thể được song song hóa. Hiệu suất sử dụng GPU/Tensor Cores phụ thuộc workload, batch size, kernels và giới hạn bộ nhớ.

### FIX-05 — Temperature=0 is not a correctness guarantee [P0]

- **File:** `Module_04_Transformer_Architecture.md` §3.6; `NCA_GENL_Exam_Guide_and_Tips.md` (sampling quick-reference).
- **Find:** claims that `temperature=0` guarantees `chính xác`, `không có tính ngẫu nhiên`, or stable outputs.
- **Replace with:**

  > Temperature điều chỉnh phân phối xác suất token. Giá trị thấp thường giảm độ đa dạng; nhiều API quy ước `temperature=0` là greedy decoding. **Greedy decoding không đảm bảo đầu ra đúng sự thật, an toàn hay giống hệt giữa mọi lần chạy**; tính tái lập còn tùy backend, precision, batching và implementation. Với code/SQL/JSON, phải kết hợp schema validation, tests, permissions và các bước xác minh khác.

- **Exam cue:** Temperature controls sampling diversity, **not** truthfulness.

### FIX-06 — Subword tokenization does not universally prevent OOV [P1]

- **File:** `Module_04_Transformer_Architecture.md` §3.1.
- **Find:** `không bao giờ bị lỗi OOV`.
- **Replace with:**

  > Subword tokenization thường làm giảm vấn đề OOV so với word-level tokenization. Các tokenizer có byte fallback/byte-level coverage có thể biểu diễn nhiều chuỗi đầu vào hơn; nhưng khả năng bao phủ chính xác phụ thuộc thiết kế tokenizer và tập ký tự hỗ trợ. Không nên hứa rằng **mọi** subword tokenizer đều không gặp OOV.

### FIX-07 — F1 is not always the best metric for imbalance [P0]

- **File:** `Module_05_Model_Selection.md` §4.1 and §6; `NCA_GENL_Exam_Guide_and_Tips.md`.
- **Find:** `F1-Score ... bắt buộc chọn` for any imbalanced dataset.
- **Replace with:**

  > Accuracy thường dễ đánh lừa trên dữ liệu mất cân bằng. Chọn metric theo **chi phí false positives và false negatives**: ưu tiên Recall khi bỏ sót positive gây hậu quả nghiêm trọng, Precision khi false alarm tốn kém; F1 khi cần cân bằng Precision–Recall; PR-AUC hữu ích khi đánh giá lớp positive hiếm qua nhiều thresholds. ROC-AUC cũng hữu ích nhưng có thể cho cảm giác quá lạc quan trong một số bài toán positive cực hiếm.

- **Example:** fraud screening may prioritize recall subject to a bounded false-positive workload.

### FIX-08 — Cosine similarity is not end-to-end RAG correctness [P0]

- **File:** `Module_05_Model_Selection.md` §4.3; `Module_07_Model_Training_and_Lifecycle.md` §5.
- **Find:** `RAG: Cosine Similarity` as the overall evaluation answer.
- **Replace with:**

  > Cosine similarity đo độ tương đồng giữa các vector embedding; nó **không trực tiếp đánh giá độ chính xác của câu trả lời cuối cùng**. Với RAG, phân tách evaluation thành: **retrieval** (Recall@k, Precision@k, MRR/nDCG), **grounding/faithfulness** (câu trả lời có được hỗ trợ bởi context không), **answer correctness/relevance**, và **system metrics** (latency, throughput, cost). Metric nào phù hợp còn phụ thuộc ground truth và task.

### FIX-09 — LoRA/PEFT percentages are implementation-dependent [P1]

- **File:** `Module_06_Model_Customization.md` §5.1.
- **Find:** `đóng băng 100%` and `chỉ huấn luyện < 1%` as properties of all PEFT.
- **Replace with:**

  > PEFT là họ kỹ thuật chỉ điều chỉnh **một phần nhỏ** tham số hoặc thêm tham số phụ trợ để giảm chi phí fine-tuning. Với LoRA điển hình, trọng số base model được đóng băng và các low-rank adapters được huấn luyện. **Tỷ lệ trainable parameters phụ thuộc model, rank và các target modules**, không mặc định dưới 1% cho mọi PEFT. Một số cấu hình có thể mở khóa thêm bias, embedding hoặc module khác.

- **Reference:** https://huggingface.co/docs/peft/index

### FIX-10 — ONNX export is not universal [P1]

- **File:** `Module_07_Model_Training_and_Lifecycle.md` §6.1.
- **Find:** description implying all PyTorch/TensorFlow/scikit-learn models export cleanly to ONNX.
- **Replace with:**

  > ONNX là định dạng biểu diễn mô hình hỗ trợ interoperability. **Khả năng export và execute phụ thuộc operator support, opset version, converters, dynamic shapes và execution provider**. Một model export thành công vẫn cần kiểm tra numerical parity, latency và compatibility trên target runtime (ONNX Runtime/TensorRT).

- **References:** https://onnx.ai/ ; https://onnxruntime.ai/docs/

### FIX-11 — MIG must be described per supported GPU SKU [P1]

- **File:** `Module_08_NVIDIA_Ecosystem.md` §2.2.
- **Find:** claim that A100/H100/B200 can be partitioned into up to seven MIG instances.
- **Assessment:** **Correct for those specific product families in supported configurations**; do not mark as incorrect. It becomes misleading only when generalized to every NVIDIA GPU.
- **Append:**

  > Số lượng MIG instances tối đa **phụ thuộc model GPU và MIG profiles**. Ví dụ A100, H100, B200 có thể hỗ trợ tối đa 7 instances; các sản phẩm khác có giới hạn nhỏ hơn. MIG chỉ hoạt động trên GPU được NVIDIA hỗ trợ, không áp dụng mặc định cho toàn bộ GPU NVIDIA.

- **Official SKU table:** https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html

### FIX-12 — Confidential Computing has a defined threat model [P1]

- **File:** `Module_09_Ethical_AI.md` §3; `Module_01_AI_Infrastructure.md` §6.
- **Find:** `bảo vệ tuyệt đối`, `100%`, `tuyệt đối không thể ...`.
- **Replace with:**

  > NVIDIA Confidential Computing giúp tăng khả năng bảo vệ dữ liệu và model khi đang được xử lý (**data in use**) thông qua các cơ chế cô lập, mã hóa và attestation tùy nền tảng. Phạm vi bảo vệ phụ thuộc **threat model, hardware/software configuration, key management và attestation**. Nó không thay thế access controls, secure application design, audit logging, vulnerability management hay bảo đảm chống mọi tấn công.

- **Related correction:** on-premises hosting does **not** automatically imply an air-gapped system or absolute data sovereignty.

### FIX-13 — Quantization taxonomy needs refinement [P1]

- **File:** `Module_07_Model_Training_and_Lifecycle.md` §6.2.
- **Issue:** The section groups FP16/BF16 mixed-precision reduction together with PTQ/QAT integer quantization without clearly separating concepts.
- **Replace/add:**

  > `FP32 -> FP16/BF16` thường được mô tả là **reduced/mixed precision**. Quantization thường chỉ ánh xạ giá trị sang định dạng số có độ chính xác thấp hơn, đặc biệt INT8/INT4 hoặc các định dạng floating-point thấp như FP8, với quy tắc scales/calibration tùy kỹ thuật. **PTQ không phải lúc nào cũng cần calibration** (ví dụ một số weight-only quantization), và quantization có thể làm giảm chất lượng hoặc không tăng tốc trên phần cứng không có kernel hỗ trợ.

### FIX-14 — Avoid promising exact acceleration ratios [P1]

- **File:** `Module_07_Model_Training_and_Lifecycle.md` §6.2; `Module_08_NVIDIA_Ecosystem.md` §4.
- **Find:** example speeds such as `~1.8x - 2x`, `10x - 100x`, and `25 phút ... 8 giây`.
- **Replace/add:**

  > Các số liệu tăng tốc chỉ mang tính minh họa **trừ khi kèm model, dataset, GPU, CPU, batch size, software version, measurement methodology và nguồn benchmark**. Không có tỷ lệ tăng tốc phổ quát khi đổi sang RAPIDS, quantization hoặc TensorRT.

### FIX-15 — NVIDIA NIM and OpenAI API compatibility must be scoped [P1]

- **File:** `Module_08_NVIDIA_Ecosystem.md` §7.2.
- **Find:** `tương thích 100% với OpenAI API`.
- **Replace with:**

  > Nhiều NVIDIA NIM microservices cung cấp các API **tương thích một phần với các endpoint và schema kiểu OpenAI**, giúp tái sử dụng một số OpenAI SDK workflows. Khả năng tương thích **không đồng nghĩa với hỗ trợ toàn bộ endpoint, parameters, streaming modes hoặc behavior** của OpenAI API; kiểm tra tài liệu của từng NIM/version.

- **Reference:** https://docs.api.nvidia.com/nim/

### FIX-16 — The exam guide's “full length” label is incorrect [P1]

- **File:** `NCA_GENL_Exam_Guide_and_Tips.md` §3.
- **Find:** `Full-Length NCA-GENL Mock Exam` for 30 questions.
- **Replace with:** `30-Question NCA-GENL Practice Test (Short Mock Exam)`.
- **Action:** create a separate **50–60-question** timed practice set spanning all five official domains. Do not call unverified questions “real exam questions” or promise they predict actual items.

### FIX-17 — Fix broken repository links [P0 for usability]

- **File:** `README.md`.
- **Find:** any `file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/` URL.
- **Replace with:** relative GitHub Markdown paths, e.g.:

```markdown
[Module 01: AI Infrastructure](./Module_01_AI_Infrastructure.md)
[Module 02: AI/ML Fundamentals](./Module_02_AIML_Fundamentals.md)
[NCA-GENL Exam Guide](./NCA_GENL_Exam_Guide_and_Tips.md)
```

- **Action:** replace **all** local absolute paths in links and diagrams.

### FIX-18 — Avoid unverified claims of official exam frequency [P1]

- **Files:** Across modules, especially Modules 01, 04, 08, 09, 10.
- **Find:** `câu hỏi tủ`, `100% cho đề thi`, `chắc chắn`, `cực kỳ hay hỏi thi`, `bám sát 100%` where no direct official source is provided.
- **Replace style:** `Khái niệm đáng ôn theo blueprint`, `Câu hỏi luyện tập minh họa`, `Chủ đề liên quan tới domain ...`.
- **Reason:** NVIDIA publishes blueprint domains but does not guarantee the appearance of specific scenarios, diagrams or answer mappings.

## 2. Correct material that should be retained

The review does **not** recommend rewriting or removing the following fundamentals:

| Topic | Status | Why keep |
|---|---|---|
| Official domain weights 30/24/22/14/10 | Confirmed | Match NVIDIA blueprint |
| Self-attention's Q/K/V mechanism | Fundamentally correct | Core Transformer understanding |
| Encoder-only vs decoder-only vs encoder-decoder | Useful conceptual taxonomy | Common model architecture distinction |
| Causal attention masking | Correct core concept | Explains autoregressive next-token training |
| Greedy / top-k / top-p / temperature | Important and useful | Revise only overclaimed guarantees |
| RAG chunking, vector retrieval, hybrid search, reranking | Good coverage | Important practical integration patterns |
| LoRA / QLoRA as parameter-efficient methods | Correct high-level concept | Revise only universal percentage claims |
| Train/validation/test, cross-validation | Correct core concepts | Add leakage scenarios |
| CUDA, TensorRT, TensorRT-LLM, Triton, NIM | Relevant ecosystem | Add clearer roles and API limitations |
| NeMo Guardrails' five rail types | Consistent with current documentation | Input, Retrieval, Dialog, Execution, Output |
| A100/H100/B200 maximum seven MIG instances | Correct for listed supported SKUs | Add SKU caveat |

NVIDIA Guardrails reference: https://docs.nvidia.com/nemo/guardrails/latest/about-nemo-guardrails-library/rail-types

## 3. Missing or underweighted knowledge to add

The following is a **recommended curriculum improvement**, not an assertion that these exact items will appear on the exam.

### ADD-01 — Experiment design (highest priority)

Suggested new file: `Module_11_Experiment_Design_and_Data_Visualization.md`.

Include: hypotheses, independent/dependent variables, control group, baselines, randomization, confounding, sample selection, train/validation/test, reproducibility, stratification, ablation, hyperparameter tuning, and how to interpret observed differences.

**Scenario:** Two LLM prompts are compared. Model version, test set, evaluation rubric and execution conditions must be held constant or recorded before attributing differences to the prompt.

### ADD-02 — Data analysis and visualization (highest priority)

Include: descriptive statistics, distributions, missingness, outliers, class imbalance, histograms, scatter plots, box plots, confusion matrices, ROC vs PR curves, correlation vs causation, and common plot interpretation errors.

**Scenario:** A 99%-negative dataset can yield 99% accuracy from a model that never predicts the positive class.

### ADD-03 — Data leakage and preprocessing

Include: fit preprocessing on training data only, proper cross-validation pipelines, SMOTE applied only within the training fold, time-based splits, deduplication across partitions, and train/test contamination for LLM evaluation.

### ADD-04 — RAG evaluation and troubleshooting

Include:
- Retrieval: Recall@k, Precision@k, MRR, nDCG.
- Answer quality: correctness, relevance, faithfulness/grounding, citation support.
- Debugging: chunking errors, embedding mismatch, metadata filters, reranker mistakes, insufficient context, stale sources.
- Operational: access control on retrieval, prompt injection inside documents, cost/latency budgets.

### ADD-05 — Software developer integration topics

Add compact practical exercises:
1. Call an LLM inference endpoint from Python.
2. Validate structured JSON outputs and handle invalid outputs.
3. Compare a prompt-only baseline with a RAG implementation.
4. Log latency (including TTFT when available), token usage and failures.
5. Contrast prompt engineering, RAG and fine-tuning based on knowledge freshness, behavior change and cost.

### ADD-06 — Exam-focused practice

- A 50–60-question timed mock exam (60 minutes).
- Questions mapped to the **five official domains**, not just module numbers.
- Scenario questions with closely competing answers.
- Explanations for *why each distractor is wrong*.
- Separate answer keys to prevent accidental memorization.
- A mistake log: `question`, `topic`, `wrong assumption`, `correct rule`, `reference`.

## 4. Suggested implementation order

| Phase | Work | Priority | Acceptance criterion |
|---|---|---|---|
| 1 | Correct FIX-01, 02, 04, 05, 07, 08 | P0 | No misleading categorical statements remain |
| 2 | Fix README links and exam mock label | P0/P1 | All GitHub links resolve; 30-question test labeled accurately |
| 3 | Apply remaining nuance fixes | P1 | Each section states limitations and conditions |
| 4 | Add experiment design and visualization | P1 | New module includes theory, scenarios and short quiz |
| 5 | Add RAG metrics, leakage and Python integration | P1 | Worked examples and corresponding tests |
| 6 | Build 50–60-question mock exam | P2 | Timed practice aligned with official domains |
| 7 | Add sources, revision date and version notes | P2 | Material technical claims are traceable |

## 5. Guidance for an AI coding agent (Claude Code / Codex)

Use this as a repository maintenance checklist:

- [x] Search exact phrases listed in FIX-01 through FIX-18.
- [x] Replace inaccurate statements **without deleting otherwise correct surrounding explanations**.
- [x] Update all associated Mermaid diagrams, summary tables, quiz answers and README descriptions.
- [x] Do not introduce invented exam odds, “exam leaks” or unsupported official claims.
- [x] Add links to primary docs adjacent to technical claims.
- [x] Preserve Vietnamese pedagogical language and concise developer-oriented examples.
- [x] Prefer conditional language (`thường`, `tùy triển khai`, `có thể`) when behavior varies by configuration.
- [x] Flag GPU/model-specific specifications with exact SKU and documentation version.
- [x] Distinguish *verified fact*, *rule of thumb*, *illustrative example* and *exam preparation recommendation*.
- [x] Run a Markdown link checker and review rendered Mermaid diagrams.
- [x] Ensure all code blocks are syntactically valid or explicitly labeled pseudocode.
- [x] Document every changed file in a PR summary.

### Acceptance checks

```bash
# Run from repo root
grep -Rni 'file:///c:/Users/' --include='*.md' . || true
grep -RniE '100%|tuyệt đối|không bao giờ|bắt buộc|chắc chắn' --include='*.md' . || true
grep -Rni 'Full-Length' --include='*.md' . || true
```

**Note:** These commands identify review candidates; not every occurrence is wrong.

## 6. Primary references

1. NVIDIA NCA-GENL Certification, domains and exam information  
   https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/
2. NVIDIA CUDA C++ Programming Guide  
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/
3. NVIDIA MIG Supported GPUs (SKU-specific limits)  
   https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/supported-gpus.html
4. NVIDIA NeMo Guardrails — Rail Types  
   https://docs.nvidia.com/nemo/guardrails/latest/about-nemo-guardrails-library/rail-types
5. NVIDIA NIM API Documentation  
   https://docs.api.nvidia.com/nim/
6. Hugging Face PEFT Documentation  
   https://huggingface.co/docs/peft/index
7. ONNX Specification / Documentation  
   https://onnx.ai/
8. ONNX Runtime Documentation  
   https://onnxruntime.ai/docs/
9. PyTorch Reproducibility Guidance  
   https://docs.pytorch.org/docs/stable/notes/randomness.html
10. scikit-learn — Common pitfalls and data leakage  
    https://scikit-learn.org/stable/common_pitfalls.html

## 7. Review boundaries

This file records **targeted content review**, not a proof that every sentence, formula, diagram, example or question in the repository is correct. Product capabilities and NVIDIA exam information may change over time; review dates and exact software versions should be maintained. The suggestions are based on the repository content inspected on the date at the top and primary documentation where available.

**Final recommendation:** Keep the existing module structure, apply P0 corrections first, improve P1 caveats, and add focused coverage of Experiment Design and Data Analysis & Visualization before expanding further into advanced GPU infrastructure.

---

## 8. Round 2 Post-Update Review & Resolution Log (`2026-11-07`)

Following the second-round comparison between commits `78eade3` and `8a1b9fa`, all 4 remaining technical/exam findings and 2 UX enhancements have been resolved:

| ID | Priority | Finding from Round 2 Review | Resolution Applied |
| :--- | :---: | :--- | :--- |
| **R2-01** | **P0** | **50-Question Exam Answer Distribution & Distractor Quality:** All 50 correct answers in `NCA_GENL_Exam_Guide_and_Tips.md` (§4–5) were previously concentrated in `A` (18) and `B` (32) with `C = 0` and `D = 0`, and several distractors were trivially eliminable. | Re-engineered all 50 timed exam questions and all 30 practice questions with realistic technical misconceptions as distractors. Verified balanced answer keys: **50-Question Exam** (`A: 12, B: 12, C: 13, D: 13`) and **30-Question Practice** (`A: 7, B: 8, C: 8, D: 7`). |
| **R2-02** | **P0** | **BERTScore vs. Sentence-level Cosine Similarity:** `Module_05_Model_Selection.md` (§4.3) previously conflated `BERTScore` with pooling an entire sentence into a single embedding vector and taking cosine similarity. | Explicitly separated **Sentence-level Embedding Cosine Similarity** (single pooled vector per sentence/chunk) from **BERTScore** (token-level contextual embeddings + pairwise cosine similarity + greedy maximum matching to compute $P_{\text{BERT}}, R_{\text{BERT}}, F_{\text{BERT}}$) in `Module_05_Model_Selection.md` (§4.3 & §6) and `NCA_GENL_Exam_Guide_and_Tips.md`. |
| **R2-03** | **P1** | **VRAM Estimation Formula Oversimplification:** `Module_01_AI_Infrastructure.md` (§3.4) used fixed rules of thumb ("+15–20% for KV Cache" and "3–4x weights for Full Fine-Tuning"). | Added the explicit **KV Cache VRAM formula** ($\text{VRAM}_{\text{KV Cache}} = 2 \times B \times S \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times b_{\text{kv}}$) with a concrete 8B `BF16` GQA vs. MHA worked calculation, and detailed the **Full Fine-Tuning memory budget** with Mixed Precision `AdamW` ($2N$ weights $+ 2N$ gradients $+ 12N$ `FP32` master weights/moments $= 16N$ bytes static state + activations). |
| **R2-04** | **P1** | **NVIDIA DGX Hardware Generalization:** `Module_08_NVIDIA_Ecosystem.md` (§2.1) described NVIDIA DGX generically as an 8-GPU system. | Updated `Module_08_NVIDIA_Ecosystem.md` (§2.1) and `Module_01_AI_Infrastructure.md` (§4.2) with model-specific architectures: `DGX-1` (8x V100/P100), `DGX-2` (16x V100 via NVSwitch), `DGX A100 / H100 / H200 / B200` (8x baseboard GPUs), `DGX Station` (4x GPUs), and rack-scale `GB200 NVL72` (36 Grace CPUs + 72 Blackwell GPUs). |
| **R2-05** | **UX** | **Separate Questions from Answer Keys:** Prevent learners from accidentally seeing answers while solving practice questions. | Wrapped each answer in §3 (30 questions) and the entire Answer Key & Explanation section in §5 (50 questions) inside collapsible `<details><summary>` spoiler blocks. |
| **R2-06** | **UX** | **Interactive 60-Minute Timed Mock Exam Mode (English & Vietnamese):** Provide an exam simulation matching the real 60-minute English exam experience. | Added an interactive **🎯 60-Minute Timed Mock Exam Mode** in `index.html` supporting both **🇬🇧 English (Official Exam Language)** and **🇻🇳 Vietnamese**, complete with a 60:00 countdown timer, question palette, Mark-for-Review flags, and post-submission scoring by official blueprint domain (`30% / 24% / 22% / 14% / 10%`). |
