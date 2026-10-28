# 🎓 Cẩm nang Ôn thi Chứng chỉ NVIDIA NCA-GENL (Bản Chuyên Sâu Dành cho Developer)

Chào mừng bạn đến với bộ tài liệu ôn thi chứng chỉ **NVIDIA Certified Associate – Generative AI and LLMs (NCA-GENL)** đã được nghiên cứu, mở rộng chuyên sâu theo đúng **Official Exam Blueprint của NVIDIA**, trực quan hóa bằng sơ đồ **Mermaid** và giải thích theo **tư duy của Software Developer**.

---

## 📊 Tỷ trọng 5 Miền Kiến thức Chính thức trong Đề thi NCA-GENL

Theo Blueprint chính thức của NVIDIA, bài thi gồm **50–60 câu hỏi trong 60 phút**, phân bổ vào 5 miền (Domains) sau đây:

| Miền kiến thức (Official Exam Domain) | Tỷ trọng | Các Module tương ứng trong Dự án này |
| :--- | :--- | :--- |
| **1. Core Machine Learning and AI Knowledge** | **30%** | [Module 01](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_01_AI_Infrastructure.md), [Module 02](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_02_AIML_Fundamentals.md), [Module 03](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_03_Generative_AI_and_LLM.md), [Module 04](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_04_Transformer_Architecture.md), [Module 10](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_10_Additional_Topics.md) |
| **2. Software Development** | **24%** | [Module 06](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_06_Model_Customization.md), [Module 07](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_07_Model_Training_and_Lifecycle.md), [Module 08](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_08_NVIDIA_Ecosystem.md) |
| **3. Experimentation** | **22%** | [Module 04](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_04_Transformer_Architecture.md), [Module 05](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_05_Model_Selection.md), [Module 06](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_06_Model_Customization.md), [Module 07](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_07_Model_Training_and_Lifecycle.md) |
| **4. Data Analysis** | **14%** | [Module 05](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_05_Model_Selection.md), [Module 07](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_07_Model_Training_and_Lifecycle.md), [Module 08 (RAPIDS)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_08_NVIDIA_Ecosystem.md) |
| **5. Trustworthy AI** | **10%** | [Module 09](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_09_Ethical_AI.md) |

---

## 🗺️ Sơ đồ Tổng quan Toàn bộ Kiến thức Kỳ thi NCA-GENL (Big Picture)

```mermaid
flowchart TD
    subgraph Infra["TẦNG 1: HẠ TẦNG VẬT LÝ & PHẦN CỨNG (Module 1 & Module 8)"]
        M1["Module 1: AI Infrastructure<br/>• 5 lớp AI Stack & 4 trụ cột<br/>• CPU vs GPU (CUDA Cores vs Tensor Cores)<br/>• Memory Pooling, NVLink vs InfiniBand + GPUDirect RDMA<br/>• Data / Tensor / Pipeline Parallelism<br/>• 4 tầng Storage + GPUDirect Storage (GDS)<br/>• Cloud (OpEx) vs On-Premise (CapEx)"]
        M8_HW["Module 8 (Phần cứng & Quản trị Cụm):<br/>• Siêu máy tính NVIDIA DGX & Grace CPU (Arm + NVLink-C2C)<br/>• NVIDIA Base Command Manager<br/>• Lập trình song song CUDA & lệnh nvidia-smi"]
        M1 --- M8_HW
    end

    subgraph CoreAI["TẦNG 2: NỀN TẢNG TOÁN HỌC, ML/DL & TRANSFORMER (Module 2, 3, 4, 10)"]
        M2["Module 2: AI/ML Fundamentals<br/>• AI ⊃ ML ⊃ DL (Ẩn dụ Cờ vua)<br/>• Supervised, Unsupervised (PCA, K-Means), RL<br/>• Bagging (Random Forest) vs Boosting (XGBoost)<br/>• Algorithm vs Model vs Inference (Ẩn dụ Barista)"]
        M10["Module 10: Deep Learning Mechanics<br/>• Forward -> Loss -> Backprop -> Optimizer (Adam)<br/>• Vanishing Gradient & ReLU vs Sigmoid + Skip Connections<br/>• LayerNorm, Dropout, Latent Diffusion (VAE/U-Net), NER, RAD"]
        M3["Module 3: GenAI & Foundation Models<br/>• Discriminative P(Y|X) vs Generative P(X)<br/>• Foundation Models (Self-supervised, Multimodal)<br/>• Pre-training -> SFT -> Alignment (RLHF vs DPO)<br/>• Word2Vec (CBOW, Skip-gram) -> RNN -> Transformer"]
        M4["Module 4: Transformer Architecture<br/>• Pipeline 6 bước & Subword Tokenization (BPE)<br/>• Positional Encoding & Self-Attention (Query, Key, Value)<br/>• Decoding Controls: Temperature, Top-K, Top-P<br/>• 3 Biến thể: Encoder (BERT) | Decoder (GPT) | Seq2Seq (T5)"]
        
        M2 --> M10 --> M3 --> M4
    end

    subgraph Lifecycle["TẦNG 3: LỰA CHỌN, TÙY BIẾN & VÒNG ĐỜI MÔ HÌNH (Module 5, 6, 7)"]
        M5["Module 5: Model Selection & Evaluation<br/>• Quy trình 6 bước chọn Model & Model Card, SLM vs LLM<br/>• Confusion Matrix: Precision, Recall, F1-Score<br/>• Bộ 3 Metric: BLEU (Dịch) | ROUGE (Tóm tắt) | Cosine (RAG)<br/>• k-Fold Cross-Validation & A/B Testing"]
        M6["Module 6: Model Customization<br/>• Cấp 1: Prompt Engineering (Zero/Few-shot, CoT, System)<br/>• Cấp 2: RAG + Chunking Overlap + Reranker<br/>• Cấp 3: PEFT (LoRA, QLoRA 4-bit), Catastrophic Forgetting<br/>• Knowledge Distillation (Teacher -> Student)"]
        M7["Module 7: Training & Deployment Lifecycle<br/>• 5 Giai đoạn: Data Collection -> EDA/Regex/N-gram<br/>-> Training (Hyperparams, Overfitting) -> Perplexity<br/>• Chuẩn mở ONNX & Quantization (PTQ vs QAT, FP32->INT8)<br/>• Batch vs Real-time, Docker/K8s, Data Drift vs Concept Drift"]
        
        M5 --> M6 --> M7
    end

    subgraph Prod["TẦNG 4: HỆ SINH THÁI PHẦN MỀM NVIDIA & AI ĐẠO ĐỨC (Module 8 & Module 9)"]
        M8_SW["Module 8 (Phần mềm NVIDIA Full-Stack):<br/>• RAPIDS: cuDF (Pandas), cuML (Sklearn), cuPy, cuGraph, cuIO<br/>• TensorRT & TensorRT-LLM (Layer Fusion, In-flight Batching, PagedAttention)<br/>• NeMo Framework (Curator, Megatron, Customizer, Retriever, Guardrails)<br/>• Triton Inference Server (Dynamic Batching, Concurrent Execution)<br/>• NVIDIA NIM Microservices, NGC Catalog, GPU Operator"]
        M9["Module 9: Ethical & Trustworthy AI<br/>• Hallucination, Bias, Plagiarism, Direct/Indirect Prompt Injection<br/>• Confidential Computing (Hopper/Blackwell) bảo vệ Privacy<br/>• NeMo Guardrails (Colang) + Embedding Classifier bảo vệ Safety<br/>• Data Lineage, Red Teaming (Garak), Data Augmentation"]
        
        M8_SW --- M9
    end

    Infra ==> CoreAI ==> Lifecycle ==> Prod
```

---

## 📚 Danh mục 11 Tài liệu Ôn tập Chi tiết trong Dự án

* 📘 **[Module 01: AI Infrastructure (Hạ tầng AI chuyên dụng)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_01_AI_Infrastructure.md)**
* 📘 **[Module 02: AI/ML Fundamentals (Nền tảng AI/ML)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_02_AIML_Fundamentals.md)**
* 📘 **[Module 03: Generative AI & Large Language Models](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_03_Generative_AI_and_LLM.md)**
* 📘 **[Module 04: Transformer Architecture (Kiến trúc Transformer)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_04_Transformer_Architecture.md)**
* 📘 **[Module 05: Model Selection (Phương pháp Lựa chọn Mô hình)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_05_Model_Selection.md)**
* 📘 **[Module 06: Model Customization (Phương pháp Tùy biến Mô hình)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_06_Model_Customization.md)**
* 📘 **[Module 07: Model Training & Deployment Lifecycle](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_07_Model_Training_and_Lifecycle.md)**
* 📘 **[Module 08: NVIDIA AI Ecosystem (Hệ sinh thái Toàn diện NVIDIA)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_08_NVIDIA_Ecosystem.md)**
* 📘 **[Module 09: Ethical & Trustworthy AI (AI Đạo đức & Đáng tin cậy)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_09_Ethical_AI.md)**
* 📘 **[Module 10: Additional Technical Topics (Chủ đề Kỹ thuật Bổ trợ)](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/Module_10_Additional_Topics.md)**
* 🏆 **[Cẩm nang Chiến lược Phòng thi, 6 Bảng Phản xạ Nhanh & Đề Thi Thử 15 Câu](file:///c:/Users/ACER/OneDrive/Documents/NCA-GENL/NCA_GENL_Exam_Guide_and_Tips.md)**
