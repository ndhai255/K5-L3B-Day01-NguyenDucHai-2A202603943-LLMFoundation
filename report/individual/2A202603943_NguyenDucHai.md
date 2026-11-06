# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603943
- **Họ và tên:** Nguyễn Đức Hải
- **SEED** (in ở ô đầu notebook): 696

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Bộ kiểm tra tự động: **15/15 đạt** (10 TODO + 5 giá trị tính tay). Đủ 10 dự đoán `predict(...)`, ghi từ 17:23:12 đến 17:27:13, trước khi chạy thí nghiệm tương ứng.

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (số trong `lab1_report.json`, làm tròn 3 chữ số):

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.876 | 0.974 | 0.148 | 0 |
| 2. BoW + MLP (deep learning) | 0.918 | 1.000 | 0.364 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.873 | 1.000 | 98.867 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.922 | 0.974 | 12.275 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.933 | 1.000 | 24.221 | 0 |
| 4a. LLM nhỏ zero-shot | — | 1.000 | 1430.396 | 0.010 |
| 4b. LLM lớn zero-shot | — | 1.000 | 1310.833 | 0.096 |
| 4c. LLM nhỏ few-shot | — | 1.000 | 1745.630 | 0.020 |

- Hàng LLM chỉ chạy trên SUB (40 tin) nên không có F1 trên toàn bộ test; so sánh chất lượng giữa các hàng bằng cột F1 SUB 40.
- Trên SUB 40 tin, 1 tin sai làm F1 đổi khoảng 0.026 (0.974 ↔ 1.000), nên chênh lệch dưới ~0.05 coi như hoà.
- Đường Pareto (1 − F1 SUB theo p95): **1. BoW + LogReg** và **2. BoW + MLP**.
- Đạt SLA p95 < 50 ms: **1, 2, 3b, 3c**. Không đạt: 3a (98.9 ms) và cả ba hàng LLM (1311–1746 ms).

## 3. Dự đoán và thực tế

**(1) `tradeoff` — đường Pareto và SLA 50 ms**
- Đoán: 1. BoW + LogReg và 3b. Encoder + LogReg nằm trên Pareto; LLM và 3a kNN vượt SLA.
- Thực tế: phần SLA **đúng**: 3a (98.9 ms) và LLM vượt, 3b (12.3 ms) đạt. Phần Pareto **sai**: Pareto là **1 và 2**, còn 3b bị 2. BoW + MLP lấn át. BoW + MLP vừa ít lỗi hơn trên SUB (F1 1.000 so với 0.974) vừa nhanh hơn khoảng 34 lần (p95 0.36 ms so với 12.28 ms).
- Vì sao lệch: mình nghĩ embedding hiểu nghĩa thì F1 sẽ cao hơn BoW rõ rệt. Thực tế trên toàn bộ test, 3b và 2 gần như hoà (0.922 so với 0.918). Còn trên SUB 40 tin, chỉ cần 3b sai đúng 1 tin là bị BoW + MLP lấn át. Đường Pareto vẽ trên tập nhỏ rất nhạy với 1 tin sai, và lợi thế của embedding không hiện ra trên tập test cùng phân phối.

**(2) `emb_tricky` — tin nhắn "ngụy trang"**
- Đoán: Encoder + LogReg đoán đúng nhiều hơn BoW vì hiểu nghĩa.
- Thực tế: **hoà 5/6 – 5/6**, và cả hai sai cùng một tin: *"Last chance! Your account rewards expire tonight, confirm details to keep them"* (spam, cả hai đoán ham).
- Vì sao lệch: encoder đặt câu này gần các thông báo tài khoản thật, vì nghĩa bề mặt của chúng giống nhau. Đầu LogReg học từ SMS Spam Collection, nơi spam chủ yếu là kiểu "trúng thưởng, gọi số 09…", có rất ít phishing giọng ngân hàng. Hiểu nghĩa câu không có nghĩa là hiểu ý đồ lừa đảo nếu dữ liệu có nhãn không có kiểu đó.

**(3) `hard_eval` — 50 case khó**
- Đoán: mọi model tụt từ ~0.95 xuống khoảng 0.40–0.70; LLM dễ bị prompt injection lừa nếu không có prompt phòng thủ.
- Thực tế: BoW tụt **dưới** khoảng đoán (F1 0.333 và 0.389). Embedding nằm trong khoảng (0.653–0.727). LLM gần như **không tụt**: 4a zero-shot đạt F1 0.982, chỉ sai 1 tin ở nhóm spam giọng tự nhiên; 4d có phòng thủ đạt 1.000. Cả hai đúng 100% ở nhóm prompt injection.
- Vì sao lệch: mình đánh giá thấp mức phụ thuộc vào từ khoá của BoW, và đánh giá quá cao sức của câu injection với model mới. Các câu injection trong bộ này (ví dụ #35 "Ignore all previous instructions… Claim your £500 prize") vẫn là spam rất lộ, và model coi câu ra lệnh đổi nhãn là dữ liệu chứ không làm theo. Dự đoán `llm_models` cũng sai một phần vì lý do tương tự: mình đoán LLM không thắng được 3b. Thực tế LLM hoà 3b trên SUB (1.000 so với 0.974) nhưng vượt xa 3b trên 50 case khó (0.982–1.000 so với 0.680).

## 4. Phân tích 50 case khó

| Model | F1 SUB (test thường) | F1 50 case khó | Accuracy 50 case khó |
| --- | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.974 | **0.333** | 0.52 |
| 2. BoW + MLP | 1.000 | 0.389 | 0.56 |
| 3a. Emb + kNN | 1.000 | 0.727 | 0.70 |
| 3b. Emb + LogReg | 0.974 | 0.680 | 0.68 |
| 3c. Emb + MLP head | 1.000 | 0.653 | 0.66 |
| 4a. LLM nhỏ zero-shot | 1.000 | 0.982 | 0.98 |
| 4d. LLM nhỏ + prompt phòng thủ | 1.000 | 1.000 | 1.00 |

**Model tụt nhiều nhất:** BoW + LogReg, F1 giảm 0.641 (0.974 → 0.333). BoW + MLP giảm gần bằng (0.611). Hai model BoW tụt ở **các nhóm spam**:

| Nhóm (% đoán đúng) | BoW + LogReg | BoW + MLP | Emb + LogReg | LLM 4a |
| --- | ---: | ---: | ---: | ---: |
| Spam giọng tự nhiên | 10% | 0% | 30% | 90% |
| Thao túng / lừa đảo | 12% | 38% | 50% | 100% |
| Né từ khoá | 17% | 0% | 83% | 100% |
| Tưởng spam nhưng không phải (ham) | 100% | 100% | 90% | 100% |
| Nói về spam (ham) | 100% | 100% | 50% | 100% |

BoW gần như luôn đoán **ham** khi không thấy từ khoá spam quen thuộc. Vì vậy nó đúng 100% ở các nhóm ham nhưng bỏ sót spam (recall sập). Đây cũng là lý do precision trên tập test là 1.000 còn recall chỉ 0.779: model rất "dè dặt".

Hai case cụ thể. BoW + MLP đúng 0% ở cả hai nhóm này, nên chắc chắn sai cả hai:

- **#30 (Né từ khoá):** *"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"*. Tokenizer tách "F R E E" thành các token một chữ cái `f`, `r`, `e`. Chúng không mang trọng số spam, nên các từ mạnh nhất là "free", "entry", "txt" biến mất khỏi vector. Chỉ còn "draw" và "win", không đủ kéo xác suất qua 0.5. Embedding làm tốt hơn hẳn (83%) vì tokenizer subword và ngữ cảnh "weekly draw… to 80086" vẫn giữ được nghĩa.
- **#14 (Spam giọng tự nhiên):** *"Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back"*. Câu gần như toàn từ đời thường ("hi", "home", "day", "happy", "text me back"). Trong tập train, các từ này xuất hiện chủ yếu trong tin của bạn bè nên mang trọng số nghiêng về ham. BoW chỉ cộng trọng số từng từ rời rạc, không thấy được kịch bản "việc nhẹ lương cao". Embedding cũng chỉ đúng 30% ở nhóm này vì câu nằm gần các tin nhắn thân mật trên bản đồ nghĩa.

Ngược lại, Embedding yếu nhất ở nhóm **Nói về spam** (50%). Ví dụ #47 kể lại một tin "won £1000, call 0906" để chê: vector câu nằm gần cụm spam vì cùng chủ đề, trong khi ý định thật là cảnh báo. LLM đọc được ý định này nên đúng 100%.

## 5. Thử thách cá nhân

**Đề #1** (SEED 696 → `696 % 6 = 0`): *Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1. Báo precision/recall/F1 từng ngưỡng. Một ngân hàng sợ chặn nhầm tin OTP thì nên chọn ngưỡng nào? Vì sao?*

**Cách làm:** không train lại. Dùng lại `w`, `b` của BoW + LogReg Phần 1, tính xác suất một lần rồi đổi ngưỡng cắt:

```python
p_proba_te = predict_proba(X_te, w, b)
for thresh in [0.2, 0.5, 0.8]:
    preds_thresh = (p_proba_te >= thresh).astype(int)
    m_thresh = metrics(y_te, preds_thresh)
```

**Kết quả** (tập test 775 tin, 86 spam; TP/FP/FN suy ra từ precision và recall):

| Ngưỡng | Accuracy | Precision | Recall | F1 | TP | FP (chặn nhầm ham) | FN (lọt spam) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 0.969 | 0.844 | 0.884 | 0.864 | 76 | 14 | 10 |
| 0.5 | 0.975 | 1.000 | 0.779 | **0.876** | 67 | 0 | 19 |
| 0.8 | 0.951 | 1.000 | 0.558 | 0.716 | 48 | 0 | 38 |

**Giải thích:**
- Ngưỡng **thấp (0.2)**: model nhạy hơn với spam. Recall tăng (0.779 → 0.884, bắt thêm 9 spam) nhưng precision giảm (1.000 → 0.844), tức chặn nhầm 14 tin ham.
- Ngưỡng **cao (0.8)**: model chỉ chặn khi rất chắc. Precision giữ 1.000, nhưng recall tụt mạnh (0.779 → 0.558): lọt thêm 19 spam.
- F1 cao nhất ở 0.5, vì đây là điểm cân bằng giữa precision và recall trên tập này.

**Lựa chọn cho ngân hàng:** chọn **ngưỡng cao (0.8)**. Chặn nhầm một tin OTP (FP) làm giao dịch của khách bị gián đoạn ngay, nên chi phí của FP lớn hơn nhiều so với để lọt một tin spam (FN). Ngân hàng cần ưu tiên precision và chấp nhận recall thấp hơn.

## 6. Phương án cho sản phẩm thật

10 triệu SMS/ngày ≈ **116 tin/giây** trung bình, yêu cầu **p95 < 50 ms/tin**.

**Chọn: 3b. Encoder (MiniLM) + LogReg**, có 2. BoW + MLP làm phương án dự phòng và một lớp LLM bất đồng bộ (router, xem bên dưới).

Số liệu dẫn chứng:
- **Đạt SLA:** có 4 cách đạt: 1 (p95 0.15 ms), 2 (0.36 ms), 3b (12.3 ms), 3c (24.2 ms). Không đạt: 3a (98.9 ms) và LLM (1311–1746 ms).
- **Chất lượng trên test thường:** bốn cách gần như hoà. F1 test lần lượt là 0.876 / 0.918 / 0.922 / 0.933, và trên SUB tất cả đều 0.974–1.000.
- **Độ bền (50 case khó) là yếu tố quyết định:** 3b đạt F1 0.680, so với 0.389 của BoW + MLP. Ở nhóm né từ khoá, 3b đúng 83% còn BoW + MLP 0%. 3b cũng bền hơn 3c (0.680 so với 0.653) và nhanh hơn (12.3 ms so với 24.2 ms).
- **Thông lượng:** 3b chạy khoảng 1.100 tin/s, gấp khoảng 9,5 lần nhu cầu trung bình. Lúc cao điểm cần thêm 1–2 máy. BoW + MLP đạt 37.596 tin/s nếu cần dự phòng.
- **Chi phí:** 0 USD tiền API. Nếu dùng LLM cho toàn bộ 10 triệu tin/ngày (30 ngày/tháng): 4a khoảng 95.6 USD/ngày ≈ **2.870 USD/tháng**, 4c ≈ 6.110 USD/tháng, 4b ≈ **28.690 USD/tháng**, mà vẫn trượt SLA.

**Router đề xuất:** 3b chạy cho mọi tin. Nếu p(spam) < 0.3 hoặc > 0.7 thì trả kết quả ngay. Tin nằm trong khoảng [0.3; 0.7] thì vẫn xử lý theo kết quả 3b, đồng thời gửi sang LLM nhỏ (4a) chạy bất đồng bộ để gắn cờ hoặc thu hồi sau. LLM đạt 0.982–1.000 trên 50 case khó, nhưng p95 khoảng 1.4 s nên không thể đặt trên đường đồng bộ. Nhờ vậy độ trễ đồng bộ vẫn là của 3b. Nếu tỉ lệ tin lưỡng lự dưới 5%, chi phí LLM chỉ còn dưới 5% con số 2.870 USD/tháng. Kết quả LLM được dùng làm nhãn để train lại đầu LogReg định kỳ.

**Đánh đổi chấp nhận:**
- 3b chậm hơn BoW khoảng 34 lần và cần nhiều CPU hơn.
- p95 của 3b dao động theo phần cứng: trong một lần chạy Colab trước, cùng notebook này cho p95 khoảng 61 ms (trượt SLA). Vì vậy phải đo lại trên máy production. Nếu p95 vượt 50 ms thì chuyển sang BoW + MLP, kèm bước chuẩn hoá văn bản (bỏ dấu cách hoặc dấu chấm chen giữa chữ, đổi `0→o`, `1→i`, `€→e`) để bù điểm yếu né từ khoá.
- Rủi ro của chuẩn hoá: có thể biến tin ham bình thường thành giống spam (tăng false positive), nên phải đo lại precision sau khi thêm.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Mình đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Bắt lỗi 3 & TODO 4.2–4.3 | Vì sao `parse_label_v2` đếm sai; nhờ review `build_classification_prompt`, `parse_label`. | Giải thích "not spam" vẫn chứa "spam" khi dùng `in`; nhắc rủi ro LLM trả câu dài hoặc có Markdown. | Kiểm lại logic nhận nhãn để "not spam" ra ham; giữ hàm ngắn gọn theo đề. |
| Dự đoán Pareto & SLA | Dự đoán cách nào nằm trên Pareto, cách nào vượt SLA 50 ms. | AI đoán 1, 2, 3c và LLM nằm trên Pareto; LLM và 3a vượt SLA. | Tự viết `predict("tradeoff", ...)` theo suy luận riêng (1 và 3b), rồi đối chiếu với biểu đồ. |
| Phân tích biểu đồ đánh đổi | Hỏi về các điểm Pareto, cách bị lấn át, cách đạt SLA ít lỗi nhất, ý nghĩa SUB 40 tin. | Phân tích Pareto, chỉ ra 2. BoW + MLP lấn át 3b/3c, 1 tin sai trên SUB 40 đổi F1 khoảng 2.5 điểm. | Đối chiếu với bảng `p95`/`F1` của mình. Kết quả chạy cuối: Pareto chỉ gồm 1 và 2 (không có 4c), BoW + MLP p95 = 0.364 ms. |
| Viết báo cáo này | Nhờ Claude Code điền file báo cáo từ `lab1_report.json` và output notebook. | Điền bảng số liệu, phân tích 50 case khó, phương án sản phẩm; trình bày lại Mục 5 từ code và kết quả thử thách mình đã chạy; phát hiện thử thách làm sai đề và tên file ảnh/JSON sai. | Kiểm tra lại từng số với `lab1_report.json`; tự làm lại thử thách đề #1 trong notebook (không dùng AI). |

**Điều hiểu ra hôm nay:**
- Điểm cao trên tập test cùng phân phối không chứng minh model hiểu spam: BoW + LogReg đạt F1 0.974 trên SUB nhưng chỉ 0.333 trên 50 case khó. Muốn biết model có bền hay không thì phải có bộ test được thiết kế để đánh lừa.
- Ngưỡng quyết định là một lựa chọn nghiệp vụ, không phải hằng số. Cùng một model, đổi ngưỡng từ 0.5 lên 0.8 giữ FP = 0 nhưng làm recall tụt từ 0.779 xuống 0.558. Chọn ngưỡng nào phụ thuộc vào việc FP hay FN đắt hơn với người dùng.
- Độ trễ phụ thuộc phần cứng: cùng 3b, p95 khoảng 61 ms ở một lần chạy và 12.3 ms ở lần chạy cuối. LLM bền nhất (0.982–1.000 trên 50 case khó) nhưng p95 chậm hơn BoW + MLP khoảng 4000 lần (1430 ms so với 0.36 ms). Vì vậy cách hợp lý là kết hợp: model rẻ xử lý số đông, LLM xử lý phần khó.
