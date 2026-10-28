# Module 2: AI/ML Fundamentals (Nền tảng Trí tuệ Nhân tạo & Học máy)

> [!IMPORTANT]
> **Trọng tâm ôn thi NCA-GENL ở Module 2 (Thuộc Domain "Core ML & AI Knowledge - 30%"):**
> Phân biệt rõ mối quan hệ bao hàm **AI $\supset$ Machine Learning (ML) $\supset$ Deep Learning (DL)**; nắm vững **3 phương pháp học máy cốt lõi (Supervised, Unsupervised, Reinforcement Learning)** cùng các thuật toán kinh điển (**PCA** để giảm chiều, **K-Means** để phân cụm, **Regression vs. Classification**, **Bagging vs. Boosting**); hiểu cách các tầng ẩn (**Hidden Layers**) trích xuất đặc trưng từ thấp đến cao; nắm vững sự khác biệt giữa **Classical Programming vs. ML**, bộ ba khái niệm **Algorithm – Model – Inference**, và quy trình **7 bước huấn luyện mô hình** qua phép ẩn dụ Barista.

---

## 1. Bản chất của Trí tuệ Nhân tạo (Intelligence & AI)

* **Trí tuệ tự nhiên (Intelligence):** Là khả năng tiếp thu thông tin, hiểu biết sâu sắc và đúc kết kinh nghiệm từ những gì đã trải qua để đưa ra hành động thích hợp trong môi trường sống. Hầu hết sinh vật sống đều sở hữu trí tuệ ở các cấp độ khác nhau (con người, cá heo, tinh tinh, voi...).
* **Trí tuệ Nhân tạo (Artificial Intelligence - AI):** Là một nhánh rộng của Khoa học Máy tính nghiên cứu và xây dựng các hệ thống máy móc có khả năng thực hiện các nhiệm vụ vốn đòi hỏi trí tuệ của con người.
* **Ba năng lực cốt lõi của hệ thống AI (3 Core Capabilities):**
  1. **Discover (Khám phá):** Tự tìm ra các quy luật ẩn, thông tin và tri thức mới từ dữ liệu.
  2. **Infer (Suy luận/Phán đoán):** Đưa ra dự đoán hoặc phán đoán có cơ sở dựa trên dữ liệu mới hoặc dữ liệu chưa hoàn chỉnh.
  3. **Reason (Lập luận):** Suy luận logic nhân quả và đưa ra quyết định đánh giá tối ưu.
* **Ví dụ AI hàng ngày:** Bản đồ Google Maps định tuyến tránh kẹt xe, FaceID mở khóa điện thoại, tự động sửa lỗi chính tả (Autocorrect), thuật toán gợi ý video YouTube/TikTok hoặc sản phẩm Shopee/Amazon.

---

## 2. Phân biệt chuẩn xác: AI vs. Machine Learning vs. Deep Learning

### 2.1. Sơ đồ phân cấp mối quan hệ bao hàm
Trong đề thi NCA-GENL, bạn cần nhớ rõ: **Deep Learning là một tập con của Machine Learning, và Machine Learning là một tập con của Artificial Intelligence.**

```mermaid
flowchart TD
    subgraph AI["1. Artificial Intelligence (Trí tuệ Nhân tạo)"]
        direction TB
        DescAI["Bất kỳ hệ thống máy tính nào mô phỏng trí tuệ & hành vi con người<br/>(Bao gồm cả hệ chuyên gia Rule-based cứng như Deep Blue lẫn học máy)"]
        
        subgraph ML["2. Machine Learning (Học máy)"]
            direction TB
            DescML["Nhánh con của AI: Máy tính TỰ HỌC TỪ DỮ LIỆU<br/>mà không cần lập trình viên viết luật tường minh (Explicitly Programmed)"]
            
            subgraph DL["3. Deep Learning (Học sâu)"]
                direction TB
                DescDL["Nhánh con của ML: Sử dụng Mạng Nơ-ron Nhân tạo nhiều lớp<br/>(Multi-layered Artificial Neural Networks) xử lý dữ liệu phức tạp"]
                
                subgraph GenAI["Generative AI (AI Tạo sinh)"]
                    DescGen["Tập con của DL: Sáng tạo ra nội dung mới<br/>(Văn bản, Hình ảnh, Âm thanh, Code)"]
                end
            end
        end
    end
```

### 2.2. Phép ẩn dụ Cờ vua (The Chess Analogy - Rất dễ nhớ!)
Hãy xem cách nhân loại dạy máy tính chơi cờ vua qua 3 thế hệ:

| Cấp độ | Cách tiếp cận trong bài toán Cờ vua | Ví dụ thực tế | Bản chất kỹ thuật |
| :--- | :--- | :--- | :--- |
| **AI (Rule-based AI)** | Lập trình viên và các kiện tướng ngồi viết sẵn hàng nghìn quy tắc cứng (`if/else`) và bảng điểm vị trí quân cờ vào code. | **IBM Deep Blue (1997)** | Dựa hoàn toàn vào luật định sẵn (Hardcoded rules) và vét cạn cây tìm kiếm (Minimax). Không tự học thêm được. |
| **ML (Machine Learning)** | Không viết luật chơi chi tiết nữa; nạp vào máy **hàng triệu biên bản ván cờ lịch sử** của các đại kiện tướng để máy tự thống kê và rút ra chiến thuật. | Các engine cờ học từ cơ sở dữ liệu ván đấu | Học quy luật thống kê từ dữ liệu lịch sử (Historical Data) của con người. |
| **DL (Deep Learning)** | Chỉ dạy máy luật đi quân cơ bản, sau đó cho mạng nơ-ron nhiều lớp **tự đấu với chính mình hàng triệu ván** từ con số 0, tự sáng tạo ra những nước cờ thần sầu mà con người chưa từng nghĩ tới. | **Google DeepMind AlphaZero** | Mạng nơ-ron sâu (Deep Neural Network) kết hợp học tăng cường, tự trích xuất đặc trưng chiến thuật ở cấp độ trừu tượng cao nhất. |

---

### 2.3. Cấu trúc Deep Learning hoạt động như thế nào?
Điểm làm nên chữ **"Deep" (Sâu)** trong Deep Learning chính là việc xếp chồng nhiều **Tầng ẩn (Hidden Layers)** giữa tầng đầu vào và tầng đầu ra. Mỗi tầng ẩn sẽ tự động trích xuất các đặc trưng (**Feature Extraction**) từ mức độ chi tiết thô sơ lên mức độ trừu tượng hoàn chỉnh.

Ví dụ bài toán: Nhận diện chữ cái viết tay `"B"` từ bức ảnh đen trắng:

```mermaid
flowchart LR
    Input["Input Layer<br/>(Tầng đầu vào)<br/>Ma trận điểm ảnh 0 và 1<br/>của bức ảnh chữ 'B'"]
    
    H1["Hidden Layer 1<br/>(Low-level features)<br/>Nhận diện cạnh cơ bản:<br/>đường thẳng, kẻ ngang, góc nghiêng"]
    
    H2["Hidden Layer 2<br/>(Mid-level features)<br/>Kết hợp các cạnh thành:<br/>đường cong, nét móc, nửa vòng tròn"]
    
    H3["Hidden Layer 3<br/>(High-level features)<br/>Ghép nối các khối thành:<br/>cấu trúc chữ cái hoàn chỉnh"]
    
    Output["Output Layer<br/>(Tầng đầu ra)<br/>Phân phối xác suất:<br/>'B' = 98%, 'P' = 1.5%, 'R' = 0.5%"]

    Input --> H1 --> H2 --> H3 --> Output
```

> [!TIP]
> **Góc nhìn Developer:** Trong Machine Learning cổ điển (Traditional ML), kỹ sư phải tự làm bước **Feature Engineering** bằng tay (ví dụ tự viết code đo tỷ lệ chiều cao/chiều rộng của chữ). Trong **Deep Learning**, các tầng **Hidden Layers** tự động học cách trích xuất đặc trưng (**Automatic Feature Extraction**) hoàn toàn từ dữ liệu thô!

---

## 3. Ba Nhánh Học Máy Cốt Lõi (Supervised, Unsupervised & Reinforcement Learning)

Trong đề thi NCA-GENL (phần *Core Machine Learning - 30%*), bạn thường gặp câu hỏi phân loại bài toán và thuật toán vào đúng nhánh học máy:

```mermaid
flowchart TD
    ML["Các Phương pháp Học máy (Machine Learning Paradigms)"]
    
    subgraph Sup["1. Supervised Learning (Học có giám sát)"]
        S_Data["Dữ liệu CÓ GẮN NHÃN (Labeled Data: X -> Y)"]
        S_Reg["• Regression (Hồi quy):<br/>Dự đoán SỐ LIÊN TỤC (Giá nhà, Nhiệt độ, Doanh thu)<br/>Thuật toán: Linear Regression, SVR"]
        S_Cls["• Classification (Phân loại):<br/>Dự đoán NHÃN RỜI RẠC (Spam/Not Spam, Bệnh/Không bệnh)<br/>Thuật toán: Logistic Regression, Decision Trees, Random Forest, SVM"]
        S_Data --> S_Reg & S_Cls
    end

    subgraph Unsup["2. Unsupervised Learning (Học không giám sát)"]
        U_Data["Dữ liệu KHÔNG GẮN NHÃN (Unlabeled Data: Chỉ có X)<br/>Tự tìm cấu trúc & quy luật ẩn"]
        U_Clust["• Clustering (Phân cụm):<br/>Gom các điểm tương đồng vào K nhóm<br/>Thuật toán: K-Means, DBSCAN, Hierarchical"]
        U_Dim["• Dimensionality Reduction (Giảm chiều dữ liệu):<br/>Nén từ hàng nghìn cột xuống vài cột chính<br/>Thuật toán: PCA (Principal Component Analysis), t-SNE"]
        U_Data --> U_Clust & U_Dim
    end

    subgraph RL["3. Reinforcement Learning (Học tăng cường)"]
        R_Data["Tác tử (Agent) tương tác với Môi trường (Environment)<br/>Học qua Thử và Sai (Trial & Error) để tối đa hóa Phần thưởng (Reward)<br/>Ứng dụng: AlphaZero, Xe tự hành, RLHF cho LLM"]
    end

    ML --> Sup & Unsup & RL
```

### Làm rõ các thuật toán cổ điển hay xuất hiện trong câu hỏi trắc nghiệm:
1. **PCA (Principal Component Analysis - Phân tích Thành phần Chính):**
   * Thuộc nhánh **Unsupervised Learning (Giảm chiều dữ liệu - Dimensionality Reduction)**.
   * *Khi nào dùng?* Khi bảng dữ liệu có quá nhiều cột đặc trưng (ví dụ 1.000 features) gây chậm quá trình train và dễ bị học vẹt (Overfitting). **PCA** chiếu dữ liệu sang một không gian ít chiều hơn (ví dụ 20 cột **Principal Components**) sao cho **giữ lại được nhiều phương sai (variance / thông tin cốt lõi) nhất**, đồng thời loại bỏ nhiễu và các cột dư thừa tương quan nhau.
2. **K-Means Clustering (Phân cụm K-Means):**
   * Thuộc nhánh **Unsupervised Learning (Phân cụm)**.
   * *Khi nào dùng?* Khi bạn có danh sách 1 triệu khách hàng chưa gắn nhãn và muốn chia họ thành $K$ nhóm hành vi mua sắm khác nhau (Customer Segmentation). Thuật toán chọn $K$ tâm cụm (**Centroids**) và lặp đi lặp lại việc gom các điểm dữ liệu vào tâm gần nhất.
3. **Học Kết hợp (Ensemble Learning): Bagging vs. Boosting:**
   * **Bagging (Bootstrap Aggregating - Đại diện: `Random Forest`):** Huấn luyện nhiều cây quyết định **song song độc lập** trên các tập con dữ liệu ngẫu nhiên rồi lấy phiếu bầu đa số (Voting). Giúp **giảm phương sai (Reduce Variance) và chống Overfitting**.
   * **Boosting (Đại diện: `XGBoost`, `AdaBoost`, `Gradient Boosting`):** Huấn luyện các mô hình **nối tiếp tuần tự**, trong đó mô hình sau tập trung sửa những lỗi sai mà mô hình trước vừa mắc phải. Giúp **giảm độ chệch (Reduce Bias) và tăng độ chính xác cực cao trên dữ liệu dạng bảng**.

---

## 4. Case Study: Lập trình truyền thống vs. Machine Learning

### Bài toán kinh doanh:
Xây dựng hệ thống gợi ý sản phẩm (Recommendation System) cho sàn thương mại điện tử dựa trên lịch sử mua hàng và hồ sơ khách hàng.

```mermaid
flowchart TD
    subgraph Classical["1. Lập trình Truyền thống (Classical Programming)"]
        direction LR
        In1["Dữ liệu đầu vào (Input)<br/>+ Quy tắc cứng (Rules if/else)"] --> Engine1["Chương trình<br/>(Hardcoded Logic)"] --> Out1["Kết quả đầu ra<br/>(Answers / Gợi ý)"]
    end

    subgraph ML_Flow["2. Học máy (Machine Learning)"]
        direction LR
        In2["Dữ liệu đầu vào (Input)<br/>+ Kết quả lịch sử (Historical Answers)"] --> Engine2["Thuật toán Huấn luyện<br/>(ML Training)"] --> Out2["Mô hình chứa Quy tắc ẩn<br/>(Trained Model / Rules)"]
    end
```

* **Tiếp cận Lập trình truyền thống (Classical Programming):** Lập trình viên phải tự ngồi viết tay hàng nghìn quy tắc điều kiện (`if/else`). Khi quy mô dữ liệu lớn và hành vi người dùng thay đổi, các luật này nhanh chóng xung đột nhau, không thể bao quát hết trường hợp và cực kỳ tốn kém để bảo trì.
* **Tiếp cận Machine Learning:** Đưa lịch sử mua sắm, lượt click và giỏ hàng vào thuật toán ML để mô hình **tự động rút trích quy tắc ẩn từ dữ liệu**, tự thích ứng với hành vi theo mùa và phản hồi suy luận trong vài mili-giây.

---

## 5. Ba khái niệm cốt lõi: Algorithm, Model và Inference

Rất nhiều Developer mới làm quen với AI thường gọi lẫn lộn giữa "Thuật toán" và "Mô hình". Trong kỳ thi NCA-GENL, 3 khái niệm này được phân định cực kỳ rõ ràng:

```mermaid
flowchart LR
    Data["Tập dữ liệu huấn luyện<br/>(Training Data)"]
    Algo["1. Algorithm (Thuật toán)<br/>Công thức toán học tổng quát<br/>f(x) = w₁x₁ + w₂x₂ + b<br/>(Chưa có trọng số cụ thể)"]
    Model["2. Model (Mô hình)<br/>Sản phẩm sau khi học xong<br/>Lưu trữ bộ Trọng số (Weights)<br/>& Độ lệch (Biases) đã tối ưu"]
    Unseen["Dữ liệu mới thực tế<br/>(Unseen / Live Data)"]
    Pred["3. Inference (Suy luận)<br/>Quá trình chạy Model trên dữ liệu mới<br/>để sinh ra Dự đoán (Prediction)"]

    Data & Algo ==>|"Quá trình Training"| Model
    Model & Unseen ==>|"Chạy trên Production"| Pred
```

| Khái niệm | Bản chất kỹ thuật | Góc nhìn Developer (Tương đương trong Code) |
| :--- | :--- | :--- |
| **1. Algorithm (Thuật toán)** | Là quy trình toán học hoặc công thức logic tổng quát (ví dụ: *Linear Regression, Decision Trees, K-Means*) dùng để tìm mối tương quan giữa các biến số: $$f(x) = w_1 x_1 + w_2 x_2 + \dots + b$$ | Giống như **Source Code của một Class** (chưa khởi tạo trạng thái, mới chỉ có cấu trúc hàm). |
| **2. Model (Mô hình)** | Là **kết quả (artifact)** thu được sau khi chạy Algorithm trên một tập dữ liệu cụ thể. Mô hình chính là tệp lưu trữ các **trọng số (weights $w$)** và **độ lệch (biases $b$)** đã được tối ưu hóa. | Giống như **Compiled Binary + Database Snapshot** (ví dụ file `.pt`, `.onnx`, `.safetensors`) chứa sẵn tri thức đã học. |
| **3. Inference (Suy luận)** | Là quá trình đưa **dữ liệu mới chưa từng thấy (unseen data)** vào Model đã huấn luyện xong để lấy kết quả dự đoán. *Training chỉ diễn ra định kỳ, nhưng Inference diễn ra hàng triệu lần mỗi ngày trên Production.* | Giống như việc gọi hàm **`model.predict(new_input)`** bên trong một REST API endpoint phục vụ user. |

---

## 6. Vòng đời Huấn luyện ML qua Phép ẩn dụ Đào tạo Barista (Barista Analogy)

Để nhớ trọn vẹn vòng lặp huấn luyện một mạng nơ-ron (Forward Pass $\to$ Loss $\to$ Backpropagation $\to$ Epochs $\to$ Inference), tài liệu chuẩn của NVIDIA sử dụng phép ẩn dụ **Đào tạo một nhân viên pha chế cà phê (Barista)**:

```mermaid
flowchart TD
    S1["1. Initialize (Khởi tạo)<br/>☕ Học việc mới vào quán, chưa biết gì<br/>🤖 Khởi tạo trọng số ngẫu nhiên (Random Weights)"]
    S2["2. Guide (Hướng dẫn / Forward Pass)<br/>☕ Đưa công thức mẫu & nguyên liệu<br/>🤖 Đưa dữ liệu đầu vào đi qua thuật toán"]
    S3["3. Predict & Evaluate (Đánh giá sai số)<br/>☕ Pha thử 1 ly; Quản lý nếm thử & chấm điểm lệch vị<br/>🤖 Mô hình dự đoán; Loss Function tính độ lệch so với nhãn chuẩn"]
    S4["4. Adjust (Điều chỉnh / Backpropagation)<br/>☕ Học việc bớt đường, thêm sữa theo góp ý<br/>🤖 Lan truyền ngược cập nhật lại Weights để giảm Loss"]
    S5["5. Accelerate (Tăng tốc phần cứng)<br/>☕ Trang bị máy pha cà phê công nghiệp hiện đại<br/>🤖 Dùng GPU / Chip AI chuyên dụng tăng tốc tính toán song song"]
    S6["6. Master (Thành thạo qua nhiều Epochs)<br/>☕ Luyện tập lặp đi lặp lại nhiều ngày đến khi chuẩn vị<br/>🤖 Lặp qua nhiều chu kỳ huấn luyện (Epochs) đạt độ chính xác cao"]
    S7["7. Deploy (Triển khai thực tế / Inference)<br/>☕ Chính thức đứng quầy phục vụ khách hàng thật<br/>🤖 Đóng gói mô hình đưa lên Production chạy Inference"]

    S1 --> S2 --> S3 --> S4
    S4 -.->|"Lặp lại liên tục (Nhờ bước 5 & 6)"| S2
    S4 --> S5 --> S6 --> S7
```

---

## 7. Bẫy đề thi NCA-GENL & Câu hỏi Thực chiến (Exam Traps & Mini Quiz)

> [!WARNING]
> **Các bẫy kinh điển trong đề thi Module 2:**
> 1. **Bẫy PCA vs. K-Means:**
>    * Muốn **giảm số lượng cột đặc trưng (Dimensionality Reduction)** mà vẫn giữ lại thông tin quan trọng nhất để tăng tốc độ train $\to$ Chọn **PCA (Principal Component Analysis)**.
>    * Muốn **chia dữ liệu không gắn nhãn thành các nhóm tương đồng (Clustering)** $\to$ Chọn **K-Means**.
> 2. **Bẫy Algorithm vs. Model:** Thành phần nào lưu trữ các trọng số (weights) và độ lệch (biases) đã được tối ưu sau khi huấn luyện trên dữ liệu? $\to$ Chọn **Model** (không chọn Algorithm).
> 3. **Bẫy Loss Function vs. Backpropagation:** **Loss Function** chỉ làm nhiệm vụ *đo lường xem mô hình dự đoán sai bao nhiêu*, còn **Backpropagation** mới là cơ chế *lan truyền ngược sai số để cập nhật trọng số*.

### Câu hỏi tự kiểm tra (Mini Quiz):
1. **Câu 1:** Một nhà khoa học dữ liệu làm việc với tập dữ liệu gồm 500 biến đặc trưng (features) không gắn nhãn và muốn nén xuống còn 30 biến quan trọng nhất để trực quan hóa và giảm chi phí huấn luyện mà ít mất mát thông tin nhất. Thuật toán nào nên được sử dụng?
   * *Đáp án:* **PCA (Principal Component Analysis)**.
2. **Câu 2:** Trong mạng nơ-ron sâu (Deep Neural Network) nhận diện khuôn mặt, tầng ẩn đầu tiên (First Hidden Layer) thường đảm nhận vai trò gì?
   * *Đáp án:* Trích xuất các **đặc trưng cấp thấp (Low-level features)** như các cạnh, đường viền sáng tối, góc nghiêng cơ bản của bức ảnh.
