# Individual Reflection — Lab 18

**Tên:** Nguyễn Ngọc Hảo  
**Module phụ trách:** M1, M2, M3, M4, M5 (Toàn bộ Pipeline)

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** 
  - M1: Hierarchical Chunking và Semantic Chunking với Underthesea.
  - M2: Hybrid Search (BM25 + Dense) tích hợp Weaviate Cloud & Cohere API.
  - M3: Cross-Encoder Reranker với Jina AI API.
  - M4: Ragas 0.4.3 Evaluation với schema mapping thích ứng.
  - M5: Metadata Enrichment song song hóa bằng ThreadPoolExecutor.
- **Các hàm/class chính đã viết:**
  - `chunk_hierarchical`, `chunk_semantic` trong `m1_chunking.py`.
  - `DenseSearch`, `BM25Search`, `reciprocal_rank_fusion` trong `m2_search.py`.
  - `CrossEncoderReranker.rerank` trong `m3_rerank.py`.
  - `evaluate_ragas`, `failure_analysis` trong `m4_eval.py`.
  - `enrich_chunks` (đa luồng) trong `m5_enrichment.py`.
- **Số tests pass:** 32 / 32 tests (toàn bộ test suite).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** 
  - **Reciprocal Rank Fusion (RRF)** để hợp nhất kết quả xếp hạng từ các nguồn tìm kiếm từ vựng và vector khác nhau mà không cần chuẩn hóa điểm số.
  - **Contextual Chunk Enrichment** để mở rộng thông tin cho từng chunk nhỏ bằng cách đưa thêm ngữ cảnh cha hoặc câu hỏi giả lập (Hypothesis QA).
- **Điều bất ngờ nhất:** 
  - API Cohere Multilingual v3 cực kỳ mạnh mẽ trong việc biểu diễn ngữ nghĩa tiếng Việt, kết hợp với Reranker Jina giúp nâng cao độ chính xác đáng kể so với tìm kiếm vector đơn thuần.
  - Ragas 0.4.3 yêu cầu cấu trúc dữ liệu nghiêm ngặt và dễ gặp lỗi không tương thích với Langchain Community trên các môi trường không có GPU khi import VertexAI.
- **Kết nối với bài giảng:** 
  - Bài giảng số 18 về Production RAG, kỹ thuật tối ưu hóa tìm kiếm nâng cao (Hybrid & Rerank) và phương pháp đánh giá hệ thống RAGAS sử dụng LLM-as-a-Judge.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** 
  - Lỗi `UnexpectedStatusCodeError 429` do Weaviate Cloud Sandbox giới hạn chỉ được tạo tối đa 1 collection. Khi cả Baseline và Production chạy độc lập, chúng cố gắng tạo 2 collections khác nhau dẫn đến bị từ chối.
  - Tốc độ chạy Enrichment M5 quá chậm khi xử lý tuần tự từng câu hỏi qua LLM (108 chunks tốn hơn 15 phút).
- **Cách giải quyết:**
  - Viết cơ chế dọn dẹp (cleanup) toàn bộ collections hiện có trên Weaviate trước khi bắt đầu pha indexing mới để luôn giữ số lượng collection bằng 1.
  - Song song hóa pha Enrichment sử dụng `ThreadPoolExecutor` với 10 threads, rút ngắn thời gian xử lý từ 15 phút xuống chỉ còn 2.5 phút.
- **Thời gian debug:** Khoảng 2 giờ.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** 
  - Thiết kế hệ thống cache cho tầng Enrichment (ví dụ sử dụng SQLite hoặc JSON local) để tránh gọi API trùng lặp cho cùng một nội dung văn bản khi chạy lại pipeline nhiều lần, giúp tiết kiệm chi phí API.
- **Module nào muốn thử tiếp:** 
  - Module 1 (Chunking): Thử nghiệm sâu hơn với Agentic Chunking hoặc Layout-aware chunking đối với các tài liệu chứa cấu trúc bảng phức tạp.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5/5 |
| Code quality | 5/5 |
| Teamwork | 5/5 |
| Problem solving | 5/5 |
