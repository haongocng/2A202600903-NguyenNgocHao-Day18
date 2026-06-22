# Group Report — Lab 18: Production RAG

**Nhóm:** Nhóm 2A  
**Ngày:** 22/06/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyễn Ngọc Hảo | M1: Chunking | [x] | 8/8 |
| Nguyễn Ngọc Hảo | M2: Hybrid Search | [x] | 5/5 |
| Nguyễn Ngọc Hảo | M3: Reranking | [x] | 5/5 |
| Nguyễn Ngọc Hảo | M4: Evaluation | [x] | 4/4 |
| Nguyễn Ngọc Hảo | M5: Enrichment | [x] | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8538 | 0.8460 | -0.0078 |
| Answer Relevancy | 0.6766 | 0.6580 | -0.0185 |
| Context Precision | 0.9167 | 0.9167 | +0.0000 |
| Context Recall | 0.8250 | 0.8542 | +0.0292 |

## Key Findings

1. **Biggest improvement:** `Context Recall` tăng từ `0.8250` lên `0.8542` (+0.0292). Sự cải thiện này nhờ vào việc áp dụng Hybrid Search kết hợp BM25 (được tối ưu hóa bằng Underthesea tokenizer) và Dense Search (Sử dụng Cohere Multilingual Embedding), giúp không bỏ sót các từ khóa tiếng Việt đặc thù.
2. **Biggest challenge:** Việc giới hạn số lượng Collection của Weaviate Cloud Sandbox (tối đa 1 Collection) khiến việc chạy song song Baseline và Production dễ gặp lỗi 429/USAGE_LIMIT_EXCEEDED. Chúng tôi đã giải quyết triệt để bằng cách dọn dẹp toàn bộ collection cũ trước khi tạo collection mới.
3. **Surprise finding:** Module 5 Enrichment (contextual prepending & hypothesis QA generation) giúp tăng chất lượng chunks rõ rệt nhưng đồng thời cũng làm tăng độ dài ngữ cảnh và có thể gây nhiễu nhẹ cho LLM trong các câu hỏi đơn giản, khiến điểm `faithfulness` giảm rất nhẹ (-0.0078).

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):**
   - Trực quan hóa bảng điểm so sánh: RAG sản xuất đạt sự cân bằng tối ưu giữa Precision (91.67%) và Recall (85.42%).
2. **Biggest win — module nào, tại sao:**
   - **M5 Enrichment & M2 Hybrid Search:** Việc sinh thêm câu hỏi giả lập (Hypothesis Questions) giúp lấp đầy khoảng cách từ vựng giữa câu hỏi của người dùng và văn bản tài liệu gốc, nâng cao khả năng khớp ngữ nghĩa.
3. **Case study — 1 failure, Error Tree walkthrough:**
   - Case study về câu hỏi phê duyệt mua laptop 30 triệu: Chỉ ra sự phân mảnh của chunking thông thường đối với các tài liệu ngắn và sự cần thiết của cơ chế Parent-Child Retrieval.
4. **Next optimization nếu có thêm 1 giờ:**
   - Hiện thực hóa cơ chế **Parent-Child Retrieval / Auto-merging** trong Weaviate để giữ tính toàn vẹn của ngữ cảnh bảng biểu.
   - Thêm tính năng **Query Decomposition** để giải quyết các câu hỏi phức hợp đa nguồn.
