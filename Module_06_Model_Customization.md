# Module 6: Model Customization (Phương pháp Tùy biến Mô hình)

> [!IMPORTANT]
> **Trọng tâm ôn tập NCA-GENL ở Module 6 (Thuộc Domain "Software Development - 24%" & "Experimentation - 22%"):**
> Hiểu rõ **3 lý do doanh nghiệp cần tùy biến Foundation Model**; nắm vững thứ tự **4 cấp độ tùy biến tăng dần về chi phí và độ phức tạp** (`Prompt Engineering` $\to$ `RAG` $\to$ `Transfer Learning / Fine-Tuning` $\to$ `Training from Scratch`) cùng **Ma trận Quyết định chọn giải pháp**; phân biệt các kỹ thuật **Prompting (Zero-shot, One-shot, Few-shot, CoT, Self-Consistency, ReAct, System Prompt)**; nắm chắc luồng hoạt động, **đánh giá và gỡ lỗi (troubleshooting) hệ thống RAG** (Chunking + Overlap, Bi-Encoder Embedding, Vector DB HNSW/IVF, Hybrid Search BM25 + Vector, Cross-Encoder Reranking, `Recall@k`/`MRR`/`nDCG`, Faithfulness, ACLs); hiểu cơ chế toán học của **PEFT (LoRA, QLoRA, Adapter, Prompt/Prefix Tuning)**, hiện tượng **Catastrophic Forgetting**, **Knowledge Distillation**, và các mẫu **tích hợp LLM bằng Python cho Software Developer**.

---

## 1. Tại sao doanh nghiệp cần tùy biến mô hình?

Các mô hình nền tảng (Foundation Models như GPT-4 hay LLaMA-3 gốc) sở hữu kiến thức tổng quát cho đến thời điểm cắt dữ liệu (**Knowledge Cutoff**). Doanh nghiệp thường cần tùy biến (Customize) mô hình vì **3 lý do chính**:
1. **Thích ứng với ngôn ngữ và thuật ngữ chuyên ngành (Domain-specific Jargon & Vocabulary):** Ví dụ trong Y khoa (tên biệt dược, mã bệnh ICD-10, bệnh án lâm sàng), Luật pháp, Viễn thông hay Tài chính ngân hàng.
2. **Nâng cao hiệu năng cho tác vụ và định dạng đặc thù (Task-specific Performance & Formatting):** Đảm bảo mô hình trả về đúng cấu trúc JSON Schema nghiêm ngặt của hệ thống, sinh câu truy vấn SQL chính xác theo schema cơ sở dữ liệu công ty, hoặc tuân thủ đúng văn phong chăm sóc khách hàng của thương hiệu.
3. **Cải thiện mức độ nhận thức ngữ cảnh nội bộ & Dữ liệu thời gian thực (Internal Context & Fresh Data Awareness):** Tích hợp các chính sách nhân sự, bảng giá tồn kho thay đổi từng phút, tài liệu bảo hành sản phẩm mới ra mắt, hoặc tài liệu mật nội bộ mà mô hình công khai chưa từng được học.

---

## 2. Bốn Cấp độ Tùy biến Mô hình (Spectrum of Model Customization) & Ma trận Quyết định

Trong kỳ thi NCA-GENL, bạn cần nhớ nguyên tắc vàng của kiến trúc sư AI: **Luôn ưu tiên đi từ giải pháp đơn giản, nhanh và rẻ tiền nhất (bên trái) sang giải pháp phức tạp, đắt đỏ nhất (bên phải)!**

```mermaid
flowchart LR
    L1["Cấp độ 1:<br/>PROMPT ENGINEERING<br/>💰 Chi phí: Rất thấp (~0$)<br/>⏱️ Thời gian: Vài phút/giờ<br/>🔧 Thay đổi Weights: KHÔNG<br/>🎯 Tối ưu câu lệnh đầu vào"]
    
    L2["Cấp độ 2:<br/>RAG (Retrieval-Augmented)<br/>💰 Chi phí: Trung bình thấp<br/>⏱️ Thời gian: Vài ngày<br/>🔧 Thay đổi Weights: KHÔNG<br/>🎯 Kết nối dữ liệu nội bộ động"]
    
    L3["Cấp độ 3:<br/>TRANSFER LEARNING / FINE-TUNING<br/>💰 Chi phí: Cao<br/>⏱️ Thời gian: Vài ngày - tuần<br/>🔧 Thay đổi Weights: CÓ<br/>🎯 Học văn phong, kỹ năng & định dạng"]
    
    L4["Cấp độ 4:<br/>TRAINING FROM SCRATCH<br/>💰 Chi phí: Cực kỳ đắt đỏ (Triệu $)<br/>⏱️ Thời gian: Hàng tháng trời<br/>🔧 Thay đổi Weights: Huấn luyện mới 100%<br/>🎯 Ngôn ngữ/Miền dữ liệu mới hoàn toàn"]

    L1 ==>|"Tăng dần Chi phí GPU, Thời gian, Lượng dữ liệu & Độ phức tạp"| L2 ==> L3 ==> L4
```

### Bảng Ma trận Quyết định: Khi nào dùng Prompting vs. RAG vs. Fine-Tuning? (CỰC KỲ QUAN TRỌNG!)
Đề thi rất hay đưa ra một kịch bản doanh nghiệp và hỏi bạn nên chọn phương pháp nào. Hãy đối chiếu bảng sau:

| Tiêu chí so sánh | 1. Prompt Engineering | 2. RAG (Retrieval-Augmented Generation) | 3. Fine-Tuning (PEFT / LoRA / SFT) | 4. Training from Scratch |
| :--- | :--- | :--- | :--- | :--- |
| **Có cập nhật Trọng số (Weights) của LLM không?** | **KHÔNG** | **KHÔNG** | **CÓ** (Cập nhật Adapter LoRA hoặc toàn bộ Weights) | **CÓ** (Khởi tạo và học mới 100% từ con số 0) |
| **Giải quyết bài toán gì tốt nhất?** | Định hướng nhanh hành vi, định dạng đầu ra hoặc suy luận logic ngay trong cửa sổ ngữ cảnh. | Bổ sung **tri thức sự kiện mới (External Facts), dữ liệu nội bộ thay đổi liên tục**, giảm ảo giác và cần **trích dẫn nguồn (Citations)**. | Dạy mô hình **hành vi mới, văn phong/giọng điệu đặc thù, cấu trúc đầu ra phức tạp cố định**, hoặc ngôn ngữ chuyên ngành sâu. | Xây dựng mô hình nền tảng cho một ngôn ngữ hoàn toàn mới hoặc dữ liệu khoa học đặc biệt (DNA/Protein/Khí hậu) chưa có Base Model. |
| **Tần suất dữ liệu thay đổi** | Tức thì theo từng câu lệnh. | **Thay đổi liên tục theo từng giây/phút** (chỉ cần cập nhật Vector DB, không cần train lại model!). | Dữ liệu tĩnh, ít thay đổi (mỗi lần muốn nạp quy luật mới phải chạy lại job Fine-tune). | Dữ liệu nền tảng tĩnh (cố định tại thời điểm Knowledge Cutoff). |
| **Nguy cơ Ảo giác (Hallucination)** | Trung bình – Cao nếu hỏi kiến thức ngoài luồng. | **Thấp nhất** (vì câu trả lời bị neo chặt vào đoạn văn bản truy xuất được). | Trung bình (Fine-tuning dạy *cách trả lời* tốt hơn là dạy *nhớ chính xác sự kiện mới*). | Phụ thuộc chất lượng dữ liệu gốc. |
| **Dữ liệu cần thiết** | 0 đến vài ví dụ mẫu (Few-shot). | Kho tài liệu thô (PDF, Wiki, Word, SQL) không cần gắn nhãn. | Hàng nghìn đến hàng trăm nghìn cặp `(Instruction, Ideal Response)` đã gắn nhãn sạch. | Hàng nghìn tỷ (Trillions) tokens dữ liệu thô + Cụm hàng nghìn GPU. |

> [!TIP]
> **Góc nhìn Thực chiến:** Trong các hệ thống AI doanh nghiệp hiện đại, **RAG và Fine-Tuning KHÔNG loại trừ nhau mà thường được KẾT HỢP (RAG + Fine-Tuning)**! Ví dụ: Dùng **LoRA Fine-Tuning** để dạy mô hình hiểu thuật ngữ y khoa và xuất báo cáo chuẩn bệnh viện, đồng thời kết nối **RAG** để tra cứu hồ sơ bệnh án mới nhất của từng bệnh nhân theo thời gian thực.

---

## 3. Cấp độ 1: Prompt Engineering (Kỹ nghệ Câu lệnh)

* **Bản chất:** Là phương pháp thiết kế, cấu trúc và tối ưu hóa câu lệnh đầu vào (Input Prompt) bằng ngôn ngữ tự nhiên để dẫn dắt LLM sinh ra kết quả mong muốn thông qua cơ chế **In-Context Learning (Học trong ngữ cảnh)** mà **hoàn toàn KHÔNG làm thay đổi bất kỳ trọng số (weights) nào của mô hình**.

### 3.1. Bảng phân biệt 7 kỹ thuật Prompting cốt lõi (Rất hay hỏi trong đề thi):
| Kỹ thuật Prompting | Định nghĩa & Cách nhận diện | Ví dụ minh họa thực tế | Khi nào nên dùng? |
| :--- | :--- | :--- | :--- |
| **1. Zero-shot Prompting** | Đặt câu hỏi hoặc giao nhiệm vụ **trực tiếp mà không kèm theo bất kỳ ví dụ mẫu nào**. | *"Phân loại cảm xúc câu sau thành Tích cực hoặc Tiêu cực: 'Bộ phim này hay ngoài mong đợi.' $\to$ Đáp án:"* | Các tác vụ phổ thông, đơn giản mà LLM đã hiểu rõ bản chất từ lúc Pre-train/SFT. |
| **2. One-shot Prompting** | Cung cấp **đúng 1 ví dụ mẫu hoàn chỉnh (1 cặp Input-Output)** trước khi đưa ra câu hỏi chính. | *"Ví dụ: 'Giao hàng chậm' -> Tiêu cực. Hãy phân loại: 'Đóng gói rất kỹ' -> ?"* | Khi muốn mô hình bắt chước chính xác một định dạng đầu ra đơn giản mà không tốn nhiều token. |
| **3. Few-shot Prompting** | Cung cấp từ **2 đến $N$ ví dụ mẫu đa dạng** ngay trong prompt để dạy mô hình quy luật ánh xạ tại chỗ (**In-context learning**). | Đưa 3–5 ví dụ chuyển đổi câu nói tự nhiên thành mã JSON chuẩn của công ty trước khi yêu cầu xử lý câu mới. | Khi tác vụ có định dạng đặc thù hoặc cần phân biệt các trường hợp biên phức tạp mà chưa muốn fine-tune. |
| **4. Chain-of-Thought (CoT)** | Yêu cầu mô hình **trình bày rõ từng bước lập luận trung gian** trước khi chốt đáp án cuối cùng (*"Let's think step by step"*). | *"Lan có 5 quả táo, mua thêm 2 túi mỗi túi 3 quả... Hãy suy nghĩ từng bước một trước khi đưa ra kết quả."* | Cực kỳ hiệu quả cho bài toán **Toán học, Logic, Lập trình và Suy luận đa bước** (kết hợpได้ với cả Zero-shot CoT và Few-shot CoT). |
| **5. Self-Consistency** | Mở rộng của CoT: Cho mô hình sinh ra **nhiều luồng suy luận CoT độc lập khác nhau** (với Temperature > 0), sau đó lấy **biểu quyết đa số (Majority Vote)** cho đáp án cuối cùng. | Sinh 5 cách giải cho cùng 1 bài toán phức tạp; nếu 4/5 cách đều ra đáp số `42`, chọn `42`. | Khi bài toán suy luận cực khó, một luồng CoT đơn lẻ vẫn có nguy cơ tính nhầm ở bước giữa. |
| **6. ReAct (Reason + Act)** | Mô hình lặp đi lặp lại chu trình: **Suy nghĩ (`Thought`) $\to$ Gọi công cụ bên ngoài (`Action`: Search API, SQL, Calculator) $\to$ Quan sát kết quả (`Observation`)** cho đến khi giải xong bài toán. | *"Thought: Cần tìm giá cổ phiếu NVDA hôm nay -> Action: gọi YahooFinance API -> Observation: \$140 -> Answer..."* | Nền tảng cốt lõi để xây dựng **AI Agents & Tool Calling / Function Calling**. |
| **7. System Prompt (Role / Persona)** | Câu lệnh nền định nghĩa **vai trò (Persona), quy tắc ứng xử, giới hạn bảo mật và định dạng đầu ra** của toàn bộ cuộc hội thoại. | *"Bạn là trợ lý tài chính của ngân hàng X. Chỉ trả lời bằng tiếng Việt lịch sự về lãi suất, từ chối mọi câu hỏi chính trị..."* | Định hình hành vi tổng thể của ứng dụng; có **độ ưu tiên cao hơn User Prompt** để giữ nhất quán và chống bẻ khóa. |

---

## 4. Cấp độ 2: Retrieval-Augmented Generation (RAG - Sinh văn bản Tăng cường Truy xuất)

* **Bản chất của RAG:** Giống như cho sinh viên đi thi dưới hình thức **"Thi mở sách (Open-Book Exam)"**. Thay vì bắt mô hình học thuộc lòng mọi tài liệu nội bộ vào trọng số (Parametric Memory), hệ thống sẽ **tự động tra cứu đúng đoạn tài liệu liên quan nhất từ kho lưu trữ ngoài (Non-parametric Memory / Vector Database) rồi đính kèm trực tiếp vào Prompt** để LLM đọc và tổng hợp câu trả lời!

### 4.1. Sơ đồ Kiến trúc & Quy trình 2 Pha của Hệ thống RAG Doanh nghiệp (Chuẩn NVIDIA NeMo Retriever)

```mermaid
flowchart TD
    subgraph Offline["PHA 1: OFFLINE DATA INGESTION & INDEXING (Chuẩn bị & Lập chỉ mục tri thức)"]
        direction LR
        Docs["Tài liệu Doanh nghiệp<br/>(PDF, Word, Wiki, HTML, SQL)"] -->|1. Parsing & Cleaning| Clean["Văn bản & Bảng biểu<br/>đã làm sạch"]
        Clean -->|"2. Chunking<br/>(Cắt đoạn + Chunk Overlap)"| Chunks["Các đoạn văn nhỏ<br/>(Semantic Chunks)"]
        Chunks -->|"3. Embedding Model<br/>(Bi-Encoder: NeMo Retriever)"| Vecs["Dense Vectors<br/>(VD: 1024 chiều)"]
        Vecs -->|"4. Indexing (HNSW / IVF)"| VDB[("Vector Database<br/>(Milvus / cuVS, FAISS, Pinecone)")]
    end

    subgraph Online["PHA 2: ONLINE RETRIEVAL, RERANKING & GENERATION (Khi User đặt câu hỏi)"]
        direction LR
        User["Câu hỏi của User<br/>(User Query)"] -->|"1. Nhúng bằng CÙNG<br/>Embedding Model"| QVec["Query Vector"]
        QVec <-->|"2a. Hybrid Search (Vector Cosine + BM25)<br/>Lấy nhanh Top-25 Chunks ứng viên"| VDB
        VDB -->|"2b. Reranker (Cross-Encoder)<br/>Chấm điểm ngữ cảnh sâu chọn Top-3 chuẩn nhất"| Aug["3. Prompt Augmentation<br/>Gộp: System Prompt + Top-3 Chunks + User Query"]
        Aug -->|"4. Generation"| LLM["Mô hình LLM<br/>(Temperature thấp ~ 0.1)"]
        LLM --> Ans["Câu trả lời chính xác<br/>+ Trích dẫn nguồn (Citations)"]
    end
```

### 4.2. Giải phẫu các thành phần kỹ thuật quan trọng trong RAG:
1. **Chiến lược Chunking & Chunk Overlap (Cắt nhỏ tài liệu):**
   * *Tại sao cần Chunking?*
     1. Tránh vượt quá giới hạn **Cửa sổ ngữ cảnh (Context Window)** và giảm hiện tượng **"Lost in the Middle"** (LLM thường khai thác thông tin ở phần đầu và phần cuối prompt tốt hơn phần chìm ở giữa một văn bản rất dài).
     2. Tăng **độ tập trung ngữ nghĩa của Vector Embedding (Retrieval Precision)**: Nếu nhúng cả cuốn sách 200 trang thành 1 vector duy nhất, ý nghĩa của từng chi tiết sẽ bị trung bình hóa và pha loãng; cắt thành từng đoạn nhỏ (ví dụ `256 - 512 tokens`) giúp vector đại diện rõ nét cho từng ý cụ thể.
     3. Giảm độ trễ (**TTFT**) và tiết kiệm chi phí token mỗi lần gọi LLM.
   * *Tại sao thường dùng **Chunk Overlap** (Gối đầu khoảng 10–20%, ví dụ chunk 500 tokens gối đầu 50 tokens)?* Để **bảo toàn ngữ cảnh liên tục tại vị trí vết cắt**, tránh việc một câu văn hoặc ý tưởng quan trọng bị cắt đôi nằm nửa ở cuối Chunk 1, nửa ở đầu Chunk 2 làm mất ngữ cảnh.
2. **Cơ sở dữ liệu Vector (Vector Databases) & Thuật toán tìm kiếm ANN:**
   * Lưu trữ hàng triệu/tỷ vector nhúng cùng với dữ liệu gốc và **Metadata** (tên file, số trang, ngày tạo, quyền truy cập phòng ban RBAC).
   * Thay vì quét cạn từng vector một (Brute-force Flat Search chậm trên tập lớn), các Vector DB (như **Milvus được tăng tốc bằng NVIDIA cuVS/RAFT, FAISS, Pinecone, Weaviate, Qdrant**) sử dụng các thuật toán **ANN (Approximate Nearest Neighbors)** như **HNSW (Hierarchical Navigable Small World - đồ thị phân tầng)** hoặc **IVF (Inverted File Index - phân cụm tâm)** để tìm ra Top-K vector gần nhất theo **Cosine Similarity / Inner Product / L2** trong vài mili-giây.
3. **Hybrid Search (Tìm kiếm Lai) & Reranking (Sắp xếp lại bằng Cross-Encoder):**
   * **Hybrid Search = Dense Retrieval (Vector Embedding hiểu ngữ nghĩa) + Sparse Retrieval (BM25 / TF-IDF khớp chính xác mã sản phẩm, số hiệu hợp đồng, tên riêng)**: Kết hợp thế mạnh của cả hai để hạn chế bỏ sót tài liệu.
   * **Bi-Encoder (Embedding Model) vs. Cross-Encoder (Reranker Model):**
     * *Bước 1 (Bi-Encoder):* Nhúng độc lập câu hỏi và tài liệu để quét nhanh qua hàng triệu bản ghi trong Vector DB, lọc ra **Top-25 đến Top-50 ứng viên**.
     * *Bước 2 (Cross-Encoder Reranker - ví dụ trong **NVIDIA NeMo Retriever**):* Đưa đồng thời từng cặp `(Câu hỏi + Chunk ứng viên)` vào mạng Transformer Cross-Encoder để tính Self-Attention chéo giữa các từ của câu hỏi và tài liệu, chấm điểm mức độ liên quan sâu rồi lọc lấy **Top-3 đến Top-5 đoạn phù hợp nhất** đưa vào LLM.

### 4.3. Đánh giá, Gỡ lỗi (Troubleshooting) và Vận hành An toàn Hệ thống RAG
Để đánh giá và vận hành RAG trong thực tế sản xuất, kỹ sư cần tách bạch rõ bộ chỉ số của **khâu Truy xuất (Retrieval)** và **khâu Sinh câu trả lời (Generation)**:

| Nhóm đánh giá / Vận hành | Các chỉ số & Cơ chế cốt lõi | Triệu chứng lỗi thường gặp & Cách khắc phục (Troubleshooting) |
| :--- | :--- | :--- |
| **1. Chất lượng Truy xuất (Retrieval Evaluation)** | • **`Recall@k`:** Tỷ lệ tài liệu liên quan thực sự xuất hiện trong Top-$k$ chunks lấy ra.<br/>• **`Precision@k`:** Tỷ lệ chunks trong Top-$k$ thực sự hữu ích.<br/>• **`MRR` (Mean Reciprocal Rank):** Đánh giá vị trí xuất hiện đầu tiên của tài liệu đúng ($\frac{1}{\text{rank}}$).<br/>• **`nDCG@k`:** Đánh giá chất lượng thứ tự xếp hạng có trọng số vị trí. | • **Lỗi:** `Recall@k` thấp (không tìm thấy đoạn văn có đáp án).<br/>• **Nguyên nhân & Cách sửa:**<br/>  1. *Chunking sai:* Cắt mất tiêu đề/bảng biểu $\to$ Dùng Semantic/Hierarchical chunking, thêm tiêu đề cha vào metadata mỗi chunk.<br/>  2. *Embedding mismatch:* Dùng khác model khi index và khi query, hoặc thiếu từ khóa chuyên ngành $\to$ Kết hợp **Hybrid Search (BM25 + Dense Vector)**.<br/>  3. *Metadata filter quá chặt:* Lọc nhầm ngày/danh mục làm rỗng kết quả $\to$ Kiểm tra lại logic pre-filtering. |
| **2. Chất lượng Câu trả lời & Chống Ảo giác (Answer & Grounding)** | • **Faithfulness / Groundedness:** Câu trả lời có được hỗ trợ hoàn toàn bởi các chunks truy xuất không?<br/>• **Answer Relevance:** Câu trả lời có đi đúng trọng tâm câu hỏi không?<br/>• **Answer Correctness & Citation Accuracy:** Độ chính xác so với ground truth và trích dẫn đúng nguồn. | • **Lỗi:** Retrieval lấy đúng tài liệu (`Recall@k` cao) nhưng LLM vẫn trả lời sai hoặc bịa đặt.<br/>• **Nguyên nhân & Cách sửa:**<br/>  1. *Reranker cắt nhầm:* Top-25 có đáp án nhưng Reranker đẩy xuống dưới Top-3 $\to$ Tăng $k$ hoặc tinh chỉnh Reranker.<br/>  2. *Nhiễu ngữ cảnh / Lost in the Middle:* Đưa quá nhiều chunks rác vào prompt $\to$ Giữ Top-3..5 chunks chất lượng cao nhất.<br/>  3. *Stale sources:* Tài liệu cũ xung đột tài liệu mới $\to$ Lọc theo timestamp/version mới nhất. |
| **3. Bảo mật & Vận hành (Operational & Security)** | • **Access Control (RBAC / Document ACLs).**<br/>• **Chống Indirect Prompt Injection.**<br/>• **Ngân sách Latency (TTFT/TPOT) & Cost.** | • **Phân quyền truy cập:** Phải lọc quyền người dùng (**ACL metadata filter**) **ngay tại bước Retrieval** trước khi đưa chunks vào LLM (không để nhân viên thường truy xuất được bảng lương CEO).<br/>• **Indirect Prompt Injection:** Dùng **Retrieval Rails (NeMo Guardrails)** và phân tách rõ thẻ dữ liệu `<context>` với chỉ thị hệ thống. |

---

## 5. Cấp độ 3: Transfer Learning, Fine-Tuning (LoRA/QLoRA) & Knowledge Distillation

* **Bản chất của Transfer Learning (Học chuyển giao):** Tái sử dụng một mô hình đã được huấn luyện sẵn trên tập dữ liệu lớn (**Pre-trained Model**) làm điểm xuất phát, sau đó tinh chỉnh tiếp để thích ứng với tác vụ hoặc miền dữ liệu chuyên sâu của doanh nghiệp.
* **Phép ẩn dụ Lái xe (Driving Analogy):** Một người **đã biết lái xe ô tô con thành thạo** khi chuyển sang **học lái xe tải hạng nặng** sẽ học nhanh hơn nhiều so với người chưa từng cầm vô lăng bao giờ (vì họ đã nắm sẵn luật giao thông, cách nhìn gương, cảm giác chân ga/phanh)!
* **Lợi ích cốt lõi:** Tiết kiệm đáng kể thời gian và chi phí tính toán GPU, cần ít dữ liệu gắn nhãn hơn so với huấn luyện từ đầu (Training from scratch).

```mermaid
flowchart TD
    TL["Các phương pháp Transfer Learning & Tối ưu Mô hình"]
    
    subgraph FT["1. Fine-Tuning (Tinh chỉnh Trọng số)"]
        FullFT["Full Fine-Tuning:<br/>Mở khóa & cập nhật toàn bộ trọng số của mô hình gốc<br/>(Tốn nhiều VRAM & dễ gặp nguy cơ Catastrophic Forgetting)"]
        PEFT["PEFT (LoRA, QLoRA, P-Tuning, Adapters):<br/>Chỉ điều chỉnh một phần nhỏ tham số hoặc thêm tham số phụ trợ,<br/>giữ đóng băng phần lớn trọng số base model<br/>(Giảm mạnh VRAM cần thiết & hạn chế quên tri thức gốc)"]
    end

    subgraph KD["2. Knowledge Distillation (Chưng cất Tri thức)"]
        Teacher["👨‍🏫 Teacher Model (Mô hình Thầy giáo):<br/>Mô hình LLM lớn (VD: 70B / 340B), năng lực cao nhưng nặng nề"]
        Student["🧑‍🎓 Student Model (Mô hình Học sinh):<br/>Mô hình SLM nhỏ gọn (VD: 3B / 8B), học bắt chước phân phối xác suất (Soft Labels)<br/>hoặc học trên dữ liệu tổng hợp từ Thầy<br/>👉 Nhỏ gọn hơn, chạy nhanh trên Edge/Mobile!"]
        Teacher ==>|"Truyền dạy tri thức (Distillation)"| Student
    end

    TL --> FT & KD
```

### 5.1. Làm rõ chuyên sâu: Full Fine-Tuning vs. Họ các kỹ thuật PEFT (Parameter-Efficient Fine-Tuning)
1. **Hai rào cản lớn của Full Fine-Tuning:**
   * **Chi phí bộ nhớ VRAM lớn:** Để cập nhật toàn bộ 70 tỷ tham số của LLaMA-3 70B ở chuẩn 16-bit, ngoài `~140 GB` lưu trọng số (Weights), GPU còn phải lưu `~140 GB` cho đạo hàm (Gradients) và hàng trăm GB cho trạng thái bộ tối ưu hóa (Optimizer States của AdamW) + Activations $\to$ Đòi hỏi cụm nhiều GPU dung lượng lớn!
   * **Hiện tượng Catastrophic Forgetting (Quên thảm khốc):** Khi cập nhật trực tiếp toàn bộ trọng số gốc trên một tập dữ liệu nhỏ/hẹp của công ty, mô hình có nguy cơ làm suy giảm các kỹ năng ngôn ngữ và lập luận tổng quát đã học ở giai đoạn Pre-training.
2. **Giải pháp PEFT – Tinh chỉnh hiệu quả tham số (Tham khảo: [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)):**
   * **Bản chất của PEFT:** PEFT là họ kỹ thuật **chỉ điều chỉnh một phần nhỏ tham số hoặc thêm các tham số phụ trợ (adapters / soft prompts)** để giảm chi phí bộ nhớ và tính toán khi fine-tuning. Với cấu hình **LoRA điển hình**, trọng số base model được đóng băng và chỉ các low-rank adapters được huấn luyện. **Lưu ý:** Tỷ lệ tham số huấn luyện (`trainable parameters %`) **phụ thuộc vào kiến trúc model, bậc hạng `r` (rank), danh sách các tầng đích (`target_modules`)**, và không mặc định dưới 1% cho mọi cấu hình PEFT (một số thiết lập PEFT/LoRA có thể mở khóa thêm `bias`, tầng `embed_tokens` hoặc `lm_head` qua tham số `modules_to_save`).
   * **LoRA (Low-Rank Adaptation):**
     * *Cơ chế toán học:* Thay vì cập nhật trực tiếp toàn bộ ma trận trọng số gốc $W_0$ (kích thước $d \times k$), **LoRA đóng băng $W_0$ (`requires_grad = False`)** và gắn song song hai ma trận phân rã hạng thấp (**Low-rank decomposition matrices**) là $A$ (kích thước $r \times k$) và $B$ (kích thước $d \times r$) với bậc hạng $r \ll \min(d, k)$ (ví dụ $r = 8, 16, 32$):
       $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
       *(Khi khởi tạo, ma trận $A$ thường lấy mẫu ngẫu nhiên Gaussian còn ma trận $B$ khởi tạo bằng $0$ để tại bước đầu tiên $\Delta W = 0$, giữ nguyên hành vi mô hình gốc).*
     * *Hiệu quả bộ nhớ:* Ví dụ với một ma trận $4096 \times 4096$ ($16.77\text{ triệu}$ tham số) và $r = 8$, số tham số của cặp $(A, B)$ tại tầng đó chỉ là $(4096 \times 8) + (8 \times 4096) = 65.536$ tham số. Nhờ không phải lưu Optimizer States cho $W_0$, lượng VRAM huấn luyện giảm mạnh và hạn chế *Catastrophic Forgetting*.
     * *Gộp trọng số khi suy luận (Mergeable Weights):* Sau khi train xong, có thể cộng gộp $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$ để suy luận không phát sinh thêm độ trễ, hoặc giữ rời để phục vụ **Multi-LoRA Serving** trên Triton/NIM (1 Base Model trong VRAM phục vụ nhiều bộ Adapter LoRA khác nhau).
   * **QLoRA (Quantized LoRA):**
     * Giúp fine-tune LLM trên phần cứng giới hạn VRAM: **Lượng tử hóa mô hình gốc đã đóng băng $W_0$ xuống định dạng 4-bit (`NF4` - NormalFloat4)** kết hợp *Double Quantization* và *Paged Optimizers*, trong khi vẫn gắn và huấn luyện các thanh Adapter LoRA ở độ chính xác `BF16`/`FP16`.
   * **Các kỹ thuật PEFT khác (hỗ trợ trong Hugging Face PEFT & NVIDIA NeMo Customizer):**
     * **Prompt Tuning / P-Tuning:** Đóng băng base model, chỉ học thêm một chuỗi các vector liên tục (**Learnable Soft Prompt / Virtual Tokens**) gắn vào đầu chuỗi đầu vào.
     * **Prefix Tuning:** Gắn thêm các vector tiền tố học được vào các tầng Key và Value của khối Attention.
     * **Adapter Tuning:** Chèn thêm các lớp mạng bottleneck nhỏ nối tiếp vào bên trong từng khối Transformer.

```mermaid
flowchart LR
    subgraph LoRA_Diagram["Cơ chế Toán học của LoRA (Low-Rank Adaptation)"]
        direction BT
        Input_X["Đầu vào x<br/>(d = 4096)"]
        
        Frozen_W["❄️ Trọng số gốc W₀ (4096 × 4096)<br/>Đóng băng (Frozen)<br/>(Ở QLoRA: Nén W₀ xuống chuẩn 4-bit NF4)"]
        
        Mat_A["🔥 Ma trận A (4096 × r)<br/>Giảm chiều xuống rank r"]
        Mat_B["🔥 Ma trận B (r × 4096)<br/>Khôi phục chiều lên 4096"]
        
        Sum(("Cộng (+)<br/>h = W₀x + (α/r)B·A·x"))
        Output_H["Đầu ra h<br/>(d = 4096)"]

        Input_X --> Frozen_W --> Sum
        Input_X --> Mat_A -->|"Nút cổ chai hạng thấp (rank r)<br/>Chỉ cập nhật A và B (tỷ lệ params tùy r & target_modules)"| Mat_B --> Sum
        Sum --> Output_H
    end
```

### 5.2. Knowledge Distillation (Chưng cất tri thức) & Đánh giá sau Fine-Tuning
* **Knowledge Distillation:** Quá trình chuyển giao tri thức từ một mô hình lớn, phức tạp (**Teacher Model**, ví dụ LLaMA-3 70B/405B) sang một mô hình nhỏ gọn hơn (**Student Model**, ví dụ SLM 3B/8B).
  * Thay vì chỉ học trên nhãn cứng (Hard labels 0 hoặc 1), Student Model học cách bắt chước **phân phối xác suất đầu ra (Soft Labels / Logits)** hoặc học trên tập dữ liệu tổng hợp (**Synthetic Data**) do Teacher Model sinh ra $\to$ Giúp thu được mô hình nhỏ gọn hơn, giảm chi phí và độ trễ khi chạy trên thiết bị biên (Edge/Mobile) hoặc sản xuất.
* **Đánh giá mô hình sau khi Fine-Tuning:** Cần theo dõi **Validation Set Loss / Task Metrics** (trên tập kiểm định chưa từng thấy trong lúc train) để đánh giá khả năng tổng quát hóa và phát hiện sớm **Overfitting**, đồng thời kiểm tra trên bộ benchmark nền để đảm bảo mô hình không bị *Catastrophic Forgetting*.

---

## 6. Thực hành Tích hợp LLM cho Software Developer (Python Integration Patterns)

Miền kiến thức **Software Development (24%)** và **Experimentation (22%)** trong NCA-GENL nhấn mạnh việc tích hợp LLM vào ứng dụng thực tế bằng Python. Dưới đây là 4 mẫu kỹ thuật cốt lõi dành cho Developer:

### 6.1. Gọi LLM Endpoint, Đo Độ trễ (TTFT & Total Latency) và Theo dõi Token Usage
Khi tích hợp một dịch vụ suy luận (như NVIDIA NIM hoặc OpenAI-compatible API), kỹ sư cần thiết lập `timeout`, đo thời gian phản hồi token đầu tiên (**TTFT**) khi dùng `stream=True`, tổng thời gian xử lý và ghi log lượng token tiêu thụ:

```python
import time
import logging
from openai import OpenAI, APITimeoutError, APIConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_client")

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Endpoint NVIDIA NIM hoặc OpenAI-compatible server
    api_key="nvapi-placeholder-key",
    timeout=15.0,                         # Luôn đặt timeout rõ ràng cho môi trường Production
    max_retries=2
)

def call_llm_with_telemetry(prompt: str, model: str = "meta/llama-3.1-8b-instruct") -> str:
    start_time = time.perf_counter()
    ttft = None
    chunks = []

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
            stream=True,
            stream_options={"include_usage": True}
        )
        usage_info = None
        for event in stream:
            if ttft is None and event.choices and event.choices[0].delta.content:
                ttft = (time.perf_counter() - start_time) * 1000.0  # ms
            if event.choices and event.choices[0].delta.content:
                chunks.append(event.choices[0].delta.content)
            if getattr(event, "usage", None):
                usage_info = event.usage

        total_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            "status=success model=%s ttft_ms=%.1f total_ms=%.1f prompt_tokens=%s completion_tokens=%s",
            model,
            ttft or -1.0,
            total_ms,
            getattr(usage_info, "prompt_tokens", "N/A"),
            getattr(usage_info, "completion_tokens", "N/A"),
        )
        return "".join(chunks)

    except (APITimeoutError, APIConnectionError) as exc:
        total_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error("status=error error_type=%s total_ms=%.1f detail=%s", type(exc).__name__, total_ms, exc)
        raise
```

### 6.2. Xác thực Đầu ra JSON Có cấu trúc (Structured Output Validation với Pydantic)
Không bao giờ tin tưởng tuyệt đối rằng LLM (kể cả khi `temperature=0`) sẽ luôn trả về JSON hợp lệ. Luôn kiểm tra cú pháp và ràng buộc kiểu dữ liệu bằng `Pydantic` hoặc `jsonschema`:

```python
import json
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class SupportTicketAnalysis(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(min_length=5, max_length=300)
    requires_human_escalation: bool

def parse_and_validate_ticket(raw_llm_output: str) -> SupportTicketAnalysis | None:
    try:
        data = json.loads(raw_llm_output)
        return SupportTicketAnalysis.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as err:
        # Xử lý an toàn: ghi log lỗi, kích hoạt retry kèm thông báo lỗi schema, hoặc chuyển sang fallback
        logging.warning("Invalid structured output from LLM: %s", err)
        return None
```

### 6.3. So sánh Thực nghiệm: Prompt-Only Baseline vs. RAG Pipeline
Khi đánh giá xem việc bổ sung RAG có thực sự cải thiện chất lượng câu trả lời so với chỉ dùng Prompt (Prompt-only baseline) hay không, cần giữ **cùng một phiên bản model, cùng tham số giải mã (`temperature`, `max_tokens`) và cùng tập câu hỏi kiểm thử (Golden Test Set)**:

```python
def build_prompt_only(question: str) -> str:
    return f"Hãy trả lời chính xác câu hỏi sau:\nCâu hỏi: {question}\nTrả lời:"

def build_rag_prompt(question: str, retrieved_chunks: list[str]) -> str:
    context_block = "\n---\n".join(retrieved_chunks)
    return (
        "Chỉ sử dụng thông tin trong phần <context> dưới đây để trả lời câu hỏi. "
        "Nếu <context> không có thông tin, hãy trả lời 'Không đủ thông tin trong tài liệu'.\n"
        f"<context>\n{context_block}\n</context>\n"
        f"Câu hỏi: {question}\nTrả lời:"
    )
```

---

## 7. Lưu ý Ôn tập & Câu hỏi Luyện tập Minh họa (Review Notes & Practice Quiz)

> [!WARNING]
> **Các điểm cần lưu ý ở Module 6:**
> 1. **So sánh RAG vs. Fine-Tuning:**
>    * Bài toán yêu cầu *"kết nối kho tài liệu nội bộ thay đổi thường xuyên, giảm ảo giác (hallucination), kèm trích dẫn nguồn (citations) mà không cập nhật trọng số mô hình"* $\to$ Chọn **RAG**.
>    * Bài toán yêu cầu *"dạy mô hình phong cách ngôn ngữ chuyên ngành sâu, thay đổi giọng văn hoặc tuân thủ định dạng đầu ra phức tạp cố định trên tập dữ liệu gán nhãn"* $\to$ Chọn **Fine-Tuning (LoRA/PEFT)**.
> 2. **So sánh LoRA vs. QLoRA:**
>    * Đóng băng trọng số gốc và chèn các ma trận phân rã hạng thấp (Low-rank decomposition matrices) để giảm số lượng tham số huấn luyện $\to$ **LoRA**.
>    * Kết hợp LoRA với việc lượng tử hóa mô hình nền đã đóng băng xuống **4-bit (`NF4`)** để fine-tune trên phần cứng hạn chế VRAM $\to$ **QLoRA**.
> 3. **Mô hình Teacher-Student:** Dùng mô hình lớn truyền tri thức cho mô hình nhỏ gọn hơn để triển khai tiết kiệm tài nguyên $\to$ **Knowledge Distillation**.
> 4. **Hiện tượng Catastrophic Forgetting:** Khi Full Fine-Tuning làm mô hình suy giảm các kỹ năng ngôn ngữ tổng quát ban đầu $\to$ Khắc phục bằng cách dùng **PEFT (LoRA)** hoặc trộn thêm dữ liệu tổng quát vào tập fine-tune.

### Câu hỏi tự kiểm tra (Mini Quiz):
1. **Câu 1:** Một kỹ sư muốn tinh chỉnh (fine-tune) một mô hình LLaMA-3 70B cho tác vụ y khoa nhưng chỉ có 1 máy trạm GPU với dung lượng VRAM hạn chế. Kỹ thuật nào cho phép nén mô hình gốc xuống 4-bit và chỉ huấn luyện các ma trận hạng thấp gắn kèm để tiết kiệm tối đa bộ nhớ?
   * *Đáp án:* **QLoRA (Quantized Low-Rank Adaptation)**.
2. **Câu 2:** Tại sao khi thực hiện Chunking tài liệu trong hệ thống RAG, các kỹ sư thường thiết lập một khoảng chồng lấn nhỏ (**Chunk Overlap**, ví dụ 10–15% số token) giữa hai đoạn văn liên tiếp?
   * *Đáp án:* Để **bảo toàn ngữ cảnh liên tục tại vị trí ranh giới cắt**, tránh làm đứt gãy ý nghĩa của những câu văn hoặc ý tưởng nằm vắt ngang giữa hai chunk.
3. **Câu 3:** Một tập đoàn luật muốn xây dựng chatbot tra cứu hàng chục nghìn văn bản quy phạm pháp luật mới ban hành hàng tuần. Yêu cầu bắt buộc là mọi câu trả lời của AI phải trích dẫn chính xác số hiệu điều khoản luật và tuyệt đối không được tự suy diễn sai lệch. Phương pháp tùy biến nào phù hợp nhất?
   * *Đáp án:* **RAG (Retrieval-Augmented Generation)** kết hợp cơ sở dữ liệu Vector và bộ Reranker (vì dữ liệu cập nhật hàng tuần và đòi hỏi trích dẫn nguồn chính xác, chống ảo giác).
4. **Câu 4:** Khi giải các bài toán đố toán học phức tạp hoặc lập luận logic nhiều bước mà không được phép fine-tune mô hình, kỹ thuật Prompt Engineering nào yêu cầu mô hình viết ra từng bước suy luận trung gian trước khi đưa ra câu trả lời cuối cùng?
   * *Đáp án:* **Chain-of-Thought (CoT) Prompting**.
5. **Câu 5:** Hiện tượng một mô hình ngôn ngữ lớn sau khi được Full Fine-Tuning trên tập dữ liệu câu lệnh SQL lại mất đi khả năng làm thơ và trò chuyện tự nhiên vốn có của Base Model được gọi là gì?
   * *Đáp án:* **Catastrophic Forgetting (Quên thảm khốc)**.
6. **Câu 6:** Trong kiến trúc RAG nâng cao (như NVIDIA NeMo Retriever), tại sao người ta thường đặt thêm một mô hình **Reranker (Cross-Encoder)** ngay sau bước tìm kiếm Vector (Bi-Encoder)?
   * *Đáp án:* Vì Vector Search (Bi-Encoder) tối ưu cho tốc độ quét nhanh qua hàng triệu bản ghi để lấy Top-25 ứng viên nhưng nén riêng rẽ câu hỏi và tài liệu; trong khi **Reranker (Cross-Encoder)** chấm điểm tương tác ngữ cảnh sâu trực tiếp giữa câu hỏi và từng đoạn văn ứng viên để lọc ra Top-3 đoạn chính xác nhất đưa vào LLM.
