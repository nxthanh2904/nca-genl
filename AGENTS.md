# Hướng dẫn cho agent làm việc với NCA-GENL

## Phạm vi và mục tiêu

Hướng dẫn này áp dụng cho toàn bộ repository. Tuân thủ yêu cầu trực tiếp của người dùng và đọc hướng dẫn riêng trong thư mục con nếu có.

Dự án cung cấp tài liệu tiếng Việt ôn chứng chỉ **NVIDIA Certified Associate – Generative AI and LLMs (NCA-GENL)**, hướng đến Software Developer / AI Engineer. Ưu tiên tính chính xác, dễ hiểu, nhất quán giữa tài liệu và khả năng ôn tập theo tình huống.

## Cấu trúc dự án

- `README.md`: giới thiệu, bản đồ kiến thức, danh mục module và nguồn tham khảo.
- `Module_01_*.md` đến `Module_11_*.md`: nội dung học chuyên sâu; đọc module liên quan trước khi sửa.
- `NCA_GENL_Exam_Guide_and_Tips.md`: hướng dẫn ôn thi, mẹo và câu hỏi luyện tập.
- `NCA_GENL_REPOSITORY_REVIEW.md`: lịch sử rà soát, các lỗi đã được nhận diện và hướng sửa. Dùng làm ngữ cảnh để tránh đưa lỗi cũ trở lại; kiểm tra trạng thái hiện tại trước khi áp dụng đề xuất cũ.
- `index.html`: trang học tập tĩnh, chứa HTML, CSS, JavaScript và nội dung tài liệu nhúng trong `DOCS_DATA`.

Hiện repository không có manifest quản lý package, hệ thống build hay bộ kiểm thử tự động. Không tự giả định có lệnh `npm run build`, `npm test` hoặc script sinh trang.

## Quy trình làm việc

1. Đọc `git status --short`, tài liệu liên quan và các vị trí tham chiếu trước khi chỉnh sửa. Giữ nguyên thay đổi có sẵn của người dùng.
2. Dùng `rg` để tìm khái niệm liên quan trong module, README, hướng dẫn thi và `index.html`.
3. Chỉnh sửa trong phạm vi yêu cầu; giữ tên file và cấu trúc hiện có trừ khi công việc cần thay đổi chúng.
4. Khi sửa kiến thức, kiểm tra những bảng, sơ đồ, ví dụ và câu hỏi dựa trên kiến thức đó. Cập nhật phần liên quan khi cần.
5. Kiểm tra diff và các liên kết sau khi sửa. Báo rõ nội dung đã đổi, cách kiểm tra và phần chưa xác minh được.

Không tự commit, push hoặc xuất bản khi người dùng chưa yêu cầu.

## Ngôn ngữ và cách biên soạn

- Viết giải thích và trao đổi với người dùng bằng tiếng Việt, trừ khi được yêu cầu khác. Giữ thuật ngữ tiếng Anh thông dụng và giải nghĩa ở lần xuất hiện đầu tiên khi cần.
- Lưu file bằng UTF-8, giữ nguyên dấu tiếng Việt. Với Windows PowerShell, đọc văn bản bằng `Get-Content -Encoding UTF8`; không chuyển mã cả file chỉ vì terminal hiển thị sai dấu.
- Bám theo cách trình bày của file đang sửa: heading, bảng so sánh, cảnh báo GitHub (`[!IMPORTANT]`, `[!TIP]`), Mermaid, công thức LaTeX và code fence có nhãn ngôn ngữ.
- Giải thích rõ khái niệm, điều kiện áp dụng và ví dụ thực tế. Dùng bảng cho so sánh, sơ đồ cho quan hệ hoặc luồng xử lý; tránh thêm hình thức trình bày không giúp việc học.
- Phân biệt kiến thức trọng tâm ôn thi với kiến thức mở rộng. Không mặc định mọi chi tiết trong repository đều thuộc đề thi chính thức.
- Với câu hỏi luyện tập, nêu đáp án và lý do; tránh câu hỏi có nhiều cách hiểu hoặc phương án đúng phụ thuộc vào giả định không được nêu.
- Giữ công thức, đơn vị và số liệu nhất quán. Gắn nhãn số liệu minh họa; không trình bày chúng như benchmark thực tế.

## Kiểm chứng kiến thức

- Ưu tiên nguồn sơ cấp: trang chứng chỉ và exam blueprint của NVIDIA, tài liệu sản phẩm chính thức, tài liệu thư viện và bài báo gốc.
- Kiểm tra nguồn hiện hành khi cập nhật thông tin có thể thay đổi: cấu trúc kỳ thi, trọng số, thời lượng, giá, tên sản phẩm, tính năng phần cứng, API hoặc phiên bản phần mềm. Ghi nguồn và ngày kiểm tra khi phù hợp.
- Các mốc thời gian và số liệu trong repository là thông tin tại thời điểm được ghi, không tự động là thông tin hiện hành.
- Nếu các nguồn chính thức mâu thuẫn, mô tả sự khác biệt và phạm vi áp dụng; không tự chọn một con số rồi trình bày như kết luận chắc chắn.
- Tránh khẳng định tuyệt đối như “luôn luôn”, “không bao giờ”, “đảm bảo” khi còn phụ thuộc mô hình, dữ liệu, phần cứng hay cấu hình.
- Phân biệt dữ kiện đã kiểm chứng, suy luận và ví dụ minh họa. Nếu không truy cập được nguồn, nói rõ giới hạn thay vì ghi “đã xác minh”.
- Ví dụ code phải thể hiện đúng ý nghĩa kỹ thuật. Với thí nghiệm ML, chú ý data leakage, train/validation/test split, baseline, seed và điều kiện so sánh công bằng.

## Đồng bộ với trang học tập

- Xem các file Markdown là nguồn biên soạn nội dung học tập. `index.html` có bản sao nhúng trong `DOCS_DATA`; sửa Markdown không tự cập nhật trang web.
- Khi thay đổi tài liệu xuất hiện trong `DOCS_DATA`, cập nhật nội dung nhúng tương ứng trong cùng công việc, trừ khi người dùng yêu cầu chỉ sửa tài liệu nguồn.
- Kiểm tra cấu trúc thực tế của `DOCS_DATA` trước khi sửa; giữ nguyên metadata và các mục không liên quan. Serialize chuỗi đúng cách để tránh hỏng dấu nháy, xuống dòng, dấu gạch chéo ngược và ký tự đặc biệt trong JavaScript/HTML.
- Không thay toàn bộ trang chỉ để cập nhật một đoạn nội dung. Không đưa công cụ build hoặc framework mới vào dự án nếu không cần cho yêu cầu.
- Khi sửa chức năng trang, giữ hoạt động của điều hướng, hiển thị Markdown, Mermaid, KaTeX, tô màu code và lưu tiến độ học bằng `localStorage`.
- Trang đang dùng thư viện và font từ CDN; kiểm tra khả năng tải tài nguyên khi đánh giá lỗi hiển thị.

## Kiểm tra trước khi hoàn tất

- Chạy `git diff --check` và xem `git diff --stat`, `git status --short` để phát hiện lỗi khoảng trắng hoặc file bị sửa ngoài phạm vi.
- Với tài liệu: kiểm tra heading, code fence, bảng, công thức, liên kết file tương đối và các tham chiếu đến mục đã đổi tên.
- Với cập nhật nội dung nhúng: đối chiếu nội dung của mục liên quan trong `DOCS_DATA` với Markdown sau khi sửa; kiểm tra cú pháp JavaScript bằng công cụ sẵn có nếu cần.
- Với thay đổi giao diện hoặc logic: kiểm tra trên trình duyệt nếu có công cụ phù hợp, gồm điều hướng, tìm kiếm, theme, tiến độ học và render nội dung liên quan.
- Chỉ chạy ví dụ code khi phù hợp với thay đổi và môi trường có đủ phụ thuộc; không cài bộ thư viện ML/GPU chỉ để kiểm tra chỉnh sửa văn bản.
- Không thêm bộ kiểm thử hoặc phụ thuộc mới chỉ cho thay đổi tài liệu nhỏ. Không tuyên bố đã kiểm tra trình duyệt, chạy code hoặc xác minh nguồn nếu chưa thực hiện.
