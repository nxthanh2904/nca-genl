# Hướng dẫn agent — NCA-GENL

Áp dụng cho toàn repository; ưu tiên yêu cầu người dùng và đọc hướng dẫn riêng trong thư mục con.

## Dự án

Tài liệu tiếng Việt ôn **NVIDIA Certified Associate – Generative AI and LLMs**, dành cho Software Developer / AI Engineer.

- `README.md`: tổng quan, danh mục module và nguồn tham khảo.
- `Module_01_*.md`–`Module_11_*.md`: nội dung học; `NCA_GENL_Exam_Guide_and_Tips.md`: hướng dẫn thi và bài tập.
- `NCA_GENL_REPOSITORY_REVIEW.md`: lịch sử rà soát; kiểm tra hiện trạng trước khi áp dụng đề xuất cũ.
- `index.html`: trang tĩnh, nhúng tài liệu trong `DOCS_DATA`. Không giả định có build script hoặc bộ kiểm thử.

## Cách làm việc

- Đọc `git status --short`, tài liệu liên quan; dùng `rg` tìm tham chiếu trước khi sửa.
- Giữ thay đổi có sẵn của người dùng; chỉ sửa đúng phạm vi, giữ tên file và cấu trúc khi có thể.
- Trao đổi bằng tiếng Việt, lưu UTF-8; PowerShell đọc bằng `Get-Content -Encoding UTF8`. Không đổi mã hóa chỉ vì terminal hiển thị sai dấu.
- Giữ phong cách hiện có: heading, bảng, Mermaid, LaTeX, cảnh báo và code fence có nhãn.
- Không commit, push hoặc xuất bản khi chưa được yêu cầu. Chỉ thêm framework/phụ thuộc khi cần cho công việc.

## Chất lượng nội dung

- Giải thích rõ khái niệm, điều kiện áp dụng và ví dụ; phân biệt trọng tâm thi với kiến thức mở rộng.
- Ưu tiên nguồn chính thức NVIDIA, tài liệu sản phẩm/thư viện và bài báo gốc. Thông tin dễ thay đổi phải kiểm tra nguồn hiện hành; ghi nguồn và ngày khi phù hợp.
- Phân biệt dữ kiện, suy luận và minh họa; nêu giới hạn hoặc mâu thuẫn nguồn, tránh khẳng định tuyệt đối thiếu điều kiện.
- Giữ công thức, đơn vị và số liệu nhất quán; gắn nhãn số liệu minh họa. Câu hỏi luyện tập cần rõ giả định, đáp án và lý do.
- Khi sửa kiến thức, cập nhật bảng, sơ đồ, ví dụ và câu hỏi liên quan. Ví dụ ML cần tránh data leakage, chia tập đúng, có baseline, seed và điều kiện so sánh công bằng.

## Đồng bộ trang học tập

- Markdown là nguồn nội dung. Khi sửa tài liệu có trong `DOCS_DATA`, đồng bộ mục tương ứng trong `index.html`, trừ khi người dùng yêu cầu chỉ sửa nguồn.
- Kiểm tra cấu trúc thực tế, serialize chuỗi đúng cách; giữ metadata và nội dung không liên quan, không thay toàn trang để sửa một đoạn.
- Khi sửa trang, giữ điều hướng, tìm kiếm, theme, render Markdown/Mermaid/KaTeX/code và tiến độ trong `localStorage`; kiểm tra CDN khi có lỗi hiển thị.

## Trước khi hoàn tất

- Chạy `git diff --check`, xem `git diff --stat`, `git status --short` và rà soát diff.
- Kiểm tra cấu trúc Markdown, công thức, liên kết tương đối và tham chiếu bị ảnh hưởng.
- Nếu sửa `DOCS_DATA`, đối chiếu với Markdown và kiểm tra cú pháp JavaScript khi cần; nếu sửa giao diện/logic, kiểm tra trình duyệt khi có công cụ.
- Chỉ chạy code/kiểm thử phù hợp với môi trường; không cài ML/GPU hoặc thêm bộ kiểm thử cho sửa tài liệu nhỏ.
- Báo ngắn gọn phần đã đổi, cách kiểm tra và giới hạn; chỉ tuyên bố những gì thực sự đã kiểm tra.
