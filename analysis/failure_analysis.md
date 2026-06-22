# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Nhóm 2A  
**Thành viên:** Nguyễn Ngọc Hảo (M1, M2, M3, M4, M5)

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8538 | 0.8460 | -0.0078 |
| Answer Relevancy | 0.6766 | 0.6580 | -0.0185 |
| Context Precision | 0.9167 | 0.9167 | +0.0000 |
| Context Recall | 0.8250 | 0.8542 | +0.0292 |

---

## Bottom-5 Failures

### #1
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got:** Từ phòng CNTT cần có: Xác nhận về cấu hình kỹ thuật trước khi đề xuất mua laptop. Về người phê duyệt: Không tìm thấy trong context.
- **Worst metric:** `answer_relevancy` (0.0)
- **Error Tree:** Output thiếu thông tin phê duyệt -> Context thiếu bảng thẩm quyền phê duyệt -> Query Match bị lệch.
- **Root cause:** Tài liệu `mua_sam.md` rất ngắn (19 dòng) nhưng bị chia nhỏ thành nhiều chunks. Hệ thống retrieve được phần "Lưu ý đặc biệt" (laptop cần CNTT xác nhận cấu hình) nhưng bỏ lỡ phần bảng "Thẩm quyền phê duyệt" ở đầu tài liệu do độ tương đồng vector của phần bảng với từ khóa "laptop/CNTT" không đủ cao.
- **Suggested fix:** Áp dụng **Parent-Child Retrieval** (hoặc auto-merging). Khi bất kỳ child chunk nào thuộc `mua_sam.md` được retrieve, ta sẽ trả về toàn bộ document `mua_sam.md` làm context cho LLM.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got:** Nhân viên có 9 năm thâm niên được 18 ngày phép... Khoảng lương trong thời gian nghỉ phép năm: Không tìm thấy thông tin trong context.
- **Worst metric:** `answer_relevancy` (0.0)
- **Error Tree:** Output thiếu phần lương -> Context thiếu thông tin thang bảng lương -> Search/Rerank không lấy được tài liệu lương.
- **Root cause:** Đây là câu hỏi phức hợp (multi-hop/hybrid query) yêu cầu thông tin từ 2 tài liệu hoàn toàn khác nhau (`nghi_phep_2024.md` và chính sách lương). Thuật toán dense/hybrid search bị bias vào từ khóa "phép năm/thâm niên", dẫn đến việc các chunks về thang bảng lương bị xếp hạng thấp và không lọt vào top 3 sau Rerank.
- **Suggested fix:** Triển khai **Query Decomposition** (phân rã câu hỏi) ở tầng Orchestration. Chia câu hỏi thành:
  1. "9 năm thâm niên nghỉ bao nhiêu ngày phép?"
  2. "Lương Senior trong khoảng nào?"
  Sau đó thực hiện tìm kiếm độc lập và gộp kết quả.

### #3
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có. Theo chính sách mật khẩu, tất cả nhân viên bắt buộc phải kích hoạt xác thực đa yếu tố (MFA) cho các tài khoản email, VPN, hệ thống nội bộ.
- **Worst metric:** `context_recall` (0.5)
- **Error Tree:** Output thiếu phần so sánh với lịch sử v1.0 -> Context thiếu file `mat_khau_v1.md` -> Reranker loại bỏ tài liệu cũ.
- **Root cause:** Rerank lọc ra các tài liệu mới nhất (`mat_khau_v2.md`) và loại bỏ tài liệu cũ đã hết hiệu lực (`mat_khau_v1.md`) để tránh gây nhiễu cho LLM. Điều này vô tình làm giảm điểm `context_recall` khi ground truth yêu cầu có thêm thông tin so sánh với chính sách cũ.
- **Suggested fix:** Cải thiện Prompt để yêu cầu LLM tìm kiếm thông tin lịch sử chính sách nếu câu hỏi hỏi chung chung, hoặc giữ lại một phần tài liệu cũ có gắn thẻ metadata `status: deprecated`.

### #4
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG. Nhân viên tuyệt đối không được tự ý xử lý malware. Phải báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline CNTT. Tự ý xử lý bị coi là vi phạm nghiêm trọng.
- **Got:** Không. Theo chính sách bảo mật, nhân viên tuyệt đối không được tự ý xử lý malware... Báo cáo trong vòng 1 giờ qua helpdesk@cty.vn hoặc hotline ext. 9999. Phát hiện malware thuộc mức Trung bình, xử lý tối đa 8 giờ...
- **Worst metric:** `faithfulness` (0.6667)
- **Error Tree:** Output thừa chi tiết phụ -> Ragas đánh giá là ảo tưởng nhẹ -> Điểm faithfulness giảm.
- **Root cause:** LLM đã trả lời rất đầy đủ và chính xác ý chính, tuy nhiên nó đã đưa thêm các thông tin phụ như "phân loại mức độ Trung bình" và "thời gian xử lý 8 giờ" từ các đoạn context khác để câu trả lời sinh động hơn. Ragas LLM evaluator coi đây là thông tin không trực tiếp trả lời câu hỏi và hạ điểm faithfulness.
- **Suggested fix:** Thiết lập prompt system chặt chẽ hơn: *"Chỉ trả lời đúng trọng tâm câu hỏi. Tuyệt đối không bổ sung các chi tiết phụ hoặc các số liệu thống kê ngoài lề không được hỏi trực tiếp."*

### #5
- **Question:** Mentor và buddy của nhân viên mới có thể là cùng một người không? Quản lý trực tiếp có thể làm mentor không?
- **Expected:** KHÔNG cho cả hai. Mentor và buddy phải là hai người khác nhau. Quản lý trực tiếp không được làm mentor hoặc buddy.
- **Got:** Dựa trên context, câu trả lời cho cả hai là Không: 1. Mentor và Buddy không thể là cùng một người... 2. Quản lý trực tiếp không được làm Mentor...
- **Worst metric:** `faithfulness` (0.375)
- **Error Tree:** Output hoàn toàn đúng -> Ragas đánh giá sai lệch -> Phép thử đánh giá bị lỗi (Evaluator noise).
- **Root cause:** Đây là trường hợp **false negative** của Ragas. LLM sinh ra câu trả lời cực kỳ chuẩn xác và trích dẫn trực tiếp từ context, nhưng do cấu hình evaluator sử dụng LLM với prompt chấm điểm chưa tối ưu hoặc do khác biệt nhỏ trong cách hành văn, Ragas đã chấm điểm faithfulness thấp bất thường.
- **Suggested fix:** Điều chỉnh nhiệt độ của LLM Evaluator trong Ragas về `0.0` và chuẩn hóa câu trả lời đầu ra của RAG ngắn gọn hơn để giảm thiểu nhiễu đánh giá.

---

## Case Study (cho presentation)

**Question chọn phân tích:**  
`Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?`

**Error Tree walkthrough:**
1. **Output đúng?** → Sai (Thiếu thông tin người phê duyệt đơn hàng 30 triệu).
2. **Context đúng?** → Sai (Chỉ lấy được phần CNTT xác nhận cấu hình kỹ thuật, không lấy được bảng định mức phê duyệt của phòng Hành chính & Tài chính).
3. **Query rewrite OK?** → OK (Query hướng đúng vào chủ đề mua sắm thiết bị).
4. **Fix ở bước:** Tối ưu hóa Chunking. Thay vì cắt nhỏ các tài liệu quy trình ngắn như `mua_sam.md`, giữ nguyên toàn bộ file làm 1 chunk hoặc liên kết các child chunk với parent document chung.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Xây dựng cơ chế **Parent-Child Retrieval**: Lưu trữ các chunks nhỏ (children) để tìm kiếm vector chính xác, nhưng khi nạp vào LLM làm ngữ cảnh thì nạp chunk cha lớn hơn (parents) hoặc toàn bộ document gốc đối với các file có kích thước nhỏ.
- Áp dụng **Query Routing** và **Query Decomposition** để phân tách các câu hỏi phức hợp thành các câu hỏi đơn trước khi gửi tới công cụ tìm kiếm.
