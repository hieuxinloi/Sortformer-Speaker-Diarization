# Streaming Speaker Diarization with Sortformer

Dự án này thực hiện việc huấn luyện tinh chỉnh (fine-tuning) mô hình **Sortformer** từ NVIDIA (kiến trúc Transformer-based encoder-labeler) cho bài toán Speaker Diarization (nhận diện "Ai nói khi nào"). 

Mô hình được tối ưu hóa cho luồng âm thanh trực tiếp (streaming) và có khả năng phân tách giọng nói của tối đa **4 người nói (4 speakers)** cùng lúc, với độ chính xác cao ngay cả trong các trường hợp có giọng nói chồng lấp.

---

## Tính năng nổi bật
*   **Độ chính xác ấn tượng**: Đạt F1 Score 93.8% và giảm 77% tỉ lệ lỗi DER (còn 2.14%) so với mô hình gốc.
*   **Mô hình mạnh mẽ**: Sử dụng Base Model `nvidia/diar_streaming_sortformer_4spk-v2` (117 triệu tham số).
*   **Công cụ toàn diện**: Cung cấp đầy đủ file từ huấn luyện (`train.py`) đến nhận diện và tách âm thanh tự động (`inference.py`).

---

## Cài đặt Môi trường (NeMo Toolkit)

Dự án yêu cầu cài đặt phần lõi **NVIDIA NeMo Toolkit** từ mã nguồn Github để đảm bảo tính tương thích và cấu hình đầy đủ nhất cho kiến trúc nhận diện giọng nói (ASR).

```bash
# Clone repository của NeMo
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo

# Cài đặt nền tảng NeMo cùng các thư viện ASR
pip install -e ".[asr]"
```
*(Yêu cầu bạn đã cài đặt sẵn PyTorch phù hợp với phiên bản CUDA của hệ thống).*

---


## Huấn luyện Mô hình (`train.py`)

Vận hành `train.py` để tiến hành tinh chỉnh model dựa trên cấu hình (config). Cốt lõi của việc fine-tuning là trỏ đường dẫn Manifest filepath của tập Train và tập Val vào hệ thống huấn luyện PyTorch Lightning.

**Ví dụ lệnh Training chi tiết:**
```bash
python train.py --exp-name sortformer_streaming_4spk_v2 --lr 1e-5 --max-epochs 12 --es-patience 3
```

*Trong quá trình Training, script sẽ tự động theo dõi `val_f1_acc` và lưu lại model tốt nhất dưới dạng file `.nemo` vào thư mục `experiments/.../version_X/checkpoints/` thông qua cơ chế Early Stopping.*

---

## Inference và Tách Audio tự động (`inference.py`)

Sử dụng `inference.py` để chạy model nhận diện phân mảnh Speaker trên một file audio bất kỳ, từ đó tự động tách xuất thành các đoạn giọng nói của từng người một cách độc lập. Đường dẫn file Model tốt nhất đã được nhúng làm mặc định trong phần lõi code thiết lập.

**1. Lệnh Inference giản lược:**
File đã được cấu hình đủ tham số ngầm định, bạn chỉ việc cung cấp duy nhất file âm thanh `.wav` đích để phân mảnh:
```bash
python inference.py "dataset/audio_can_test/cuoc_hop_4_nguoi.wav"
```

**2. Đầu ra hệ thống (Output Structure):**
Theo quy trình, hệ thống sẽ tự sinh ra thư mục con bao bọc lấy tên gốc của audio nằm trong directory `output/`:
```text
output/
└── cuoc_hop_4_nguoi/
    ├── cuoc_hop_4_nguoi_speaker_0.rttm  # RTTM ghi nhận timeline riêng biệt của Speaker 0
    ├── cuoc_hop_4_nguoi_speaker_0.wav   # Các luồng voice ghép nối duy nhất của Speaker 0
    ├── cuoc_hop_4_nguoi_speaker_1.rttm
    ├── cuoc_hop_4_nguoi_speaker_1.wav
    ├── ...
```
Thao tác tách audio này sẽ loại bỏ hoàn toàn các cấu trúc rỗng (khoảng lặng chung, vô vị) và chỉ cung cấp các track giọng nói cực kì trong sạch phục vụ hệ thống NLP kế tiếp.
