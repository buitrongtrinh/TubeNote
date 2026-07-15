# TubeNote

[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Y1PnEIHituE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Hardware](https://img.shields.io/badge/Hardware-CPU--only%20OK%20%7C%20GPU%20optional-2ea44f)](#yêu-cầu-hệ-thống)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[English](README.md) | [Tiếng Việt](README.vi.md)**

TubeNote là ứng dụng lồng tiếng video AI và hỏi-đáp video, chạy local-first. Nó
biến một video YouTube **tiếng Anh** thành video đã lồng **tiếng Việt** (ngôn
ngữ nguồn và đích cố định, không tùy chỉnh được), giữ lại phụ đề có thể chỉnh
sửa cùng metadata thời gian, và có thêm panel chat RAG để hỏi đáp về nội dung
video.

Dự án được xây dựng như một hệ thống full-stack thực tế xoay quanh việc bản địa
hoá video: lấy phụ đề, ASR dự phòng, dịch có tính tới thời lượng, tổng hợp
giọng nói (TTS), căn chỉnh giọng nói/phụ đề, giữ lại âm thanh nền, và truy xuất
lai (hybrid retrieval) trên transcript đã xử lý.

**Toàn bộ pipeline chạy được trên máy chỉ có CPU** (faster-whisper `small.en`
int8 + Supertonic TTS). GPU NVIDIA là tuỳ chọn, mở khoá đường chất lượng cao
hơn: TTS OmniVoice có voice cloning, và nếu đủ VRAM, ASR `large-v3-turbo` (tự
chọn thay cho `medium.en` khi có sẵn — nhanh hơn và chính xác hơn với chi phí
VRAM gần như tương đương, xem `docs/dubbing.md`).

![Kiến trúc chi tiết TubeNote](docs/assets/tubenote-architecture.webp)

## Demo

| Video | Link |
| --- | --- |
| Video hướng dẫn giao diện | [Xem trên YouTube](https://www.youtube.com/watch?v=YOUR_GUI_DEMO_VIDEO_ID) |
| Mẫu video đã lồng tiếng | [Xem trên YouTube](https://www.youtube.com/watch?v=Y1PnEIHituE) |

<!-- Link video hướng dẫn giao diện đang là placeholder — thay
     YOUR_GUI_DEMO_VIDEO_ID sau khi quay xong video đó. -->

**Thư viện:**

![Thư viện TubeNote](docs/assets/screenshot-library.webp)

**Bản nháp: video đã load nhưng chưa dubbing:**

![Bản nháp TubeNote](docs/assets/screenshot-drafts.webp)

**Lồng tiếng, từng bước:**

1. Trình tạo lồng tiếng — tự phát hiện phần cứng, chọn engine/giọng/chất lượng TTS:

   ![Trình tạo lồng tiếng TubeNote](docs/assets/screenshot-create.webp)

2. Dịch nội dung — copy prompt sang ChatGPT (hoặc dịch qua API) và kiểm tra các lô trước khi đưa vào TTS:

   ![Bước dịch nội dung TubeNote](docs/assets/screenshot-translate.webp)

3. Lồng tiếng — chọn giọng và chất lượng giọng, giữ nhạc nền (lựa chọn): 

   ![Bước dubbing video](docs/assets/Screenshot-dubbing.webp)  

**Trang video — player đã lồng tiếng, transcript song ngữ, tạo lại từng đoạn:**

   ![Trang video TubeNote với player và transcript](docs/assets/screenshot-watch.webp)

## TubeNote làm những gì

1. Nạp video YouTube và metadata bằng `yt-dlp`.
2. Lấy phụ đề tiếng Anh từ YouTube nếu có sẵn.
3. Dự phòng bằng ASR faster-whisper khi không có phụ đề, tách lại mốc thời
   gian từng từ thành câu tự nhiên (dựa vào khoảng lặng, dấu câu, giới hạn số từ).
4. Sinh prompt dịch có tính tới thời lượng từ các đoạn phụ đề.
5. Cho người dùng tự dịch thủ công, hoặc dịch theo lô qua API LLM.
6. Kiểm tra các lô đã dịch trước khi đưa vào TTS.
7. Sinh giọng nói tiếng Việt bằng Supertonic (CPU) hoặc OmniVoice (GPU).
8. Căn chỉnh giọng nói đã sinh và phụ đề tiếng Việt theo mốc thời gian video.
9. Tuỳ chọn tách và giữ lại âm thanh nền bằng Demucs.
10. Phát video MP4 hoàn chỉnh trong player Vidstack có điều khiển phụ đề.
11. Xây dựng index RAG lai trên phụ đề local để hỏi đáp về video.

## Điểm nổi bật

- Luồng làm việc local-first: media, phụ đề, index Chroma, bản tóm tắt, log,
  cookie và mẫu giọng được sinh ra đều nằm ngoài git, dưới các đường dẫn runtime
  đã bị ignore.
- Thiết lập theo phần cứng: nhập RAM/VRAM của máy (giá trị tự phát hiện được
  điền sẵn), TubeNote tự chọn toàn bộ bộ tham số — kích thước model Whisper,
  engine TTS, batch size OmniVoice, số thread CPU — từ các bảng tier có thể
  calibrate trong `config.yaml`; batch vẫn tự giảm nửa khi gặp CUDA OOM thay vì
  làm job thất bại, và `scripts/measure_vram.py` đo mức dùng thực tế để tinh
  chỉnh các bảng đó.
- Bảy preset ASR (tiny.en → large-v3-turbo, CPU int8 và CUDA fp16) qua
  faster-whisper/CTranslate2, với thuật toán tách lại câu dựa trên mốc thời
  gian từng từ để có ranh giới đoạn phù hợp cho TTS.
- Hai chế độ dịch:
  - Thủ công: copy prompt sang ChatGPT rồi dán kết quả đã kiểm tra lại vào.
  - API: chọn provider/model và dịch theo lô trực tiếp từ TubeNote.
- Hai engine TTS:
  - Supertonic (CPU): đường mặc định nhanh, được khớp vào khung thời gian sau
    khi sinh.
  - OmniVoice (GPU): đường chất lượng cao hơn, sinh có điều kiện theo thời
    lượng và hỗ trợ giọng tham chiếu.
- Tạo lại từng đoạn sau khi đã lồng tiếng, bao gồm cả ánh xạ cách phát âm.
- Phụ đề tiếng Việt được căn theo mốc thời gian TTS đã sinh, không chỉ theo
  phụ đề gốc thô.
- Tuỳ chọn giữ lại nền nhạc/âm thanh bằng Demucs cho video có nhạc hoặc
  tiếng nền.
- RAG lai: truy xuất dense bằng Chroma với `BAAI/bge-m3`, truy xuất sparse
  bằng BM25, và Reciprocal Rank Fusion.
- Chat RAG bắt đầu bằng cách tạo/nạp một bản tóm tắt video đã cache, sau đó
  dùng provider/model LLM đã chọn cho phiên chat.

## Công nghệ sử dụng

| Phần | Công nghệ |
| --- | --- |
| Frontend | Next.js 14, React, Vidstack, react-markdown |
| Backend | FastAPI, Python 3.11 |
| Nạp video | `yt-dlp`, phụ đề YouTube |
| ASR | faster-whisper, CTranslate2 |
| Dịch | Luồng prompt cộng với DeepSeek/OpenAI/Gemini/Anthropic qua LangChain |
| TTS | Supertonic, OmniVoice |
| Audio/video | ffmpeg, imageio-ffmpeg, SoundFile, Demucs, torch/torchaudio |
| RAG | LangChain, Chroma, `BAAI/bge-m3`, BM25, RRF |
| Job runtime | FastAPI background tasks, trạng thái job lưu trong bộ nhớ |

## Cấu trúc repository

```text
backend/
  api/                 Router FastAPI cho dubbing, video, và RAG
  llm/providers/       Adapter provider cho DeepSeek, OpenAI, Google, Anthropic
  pipeline/            Điều phối dubbing và Q&A ở tầng cao
  services/
    dubbing/           Kiểm tra TTS, khớp thời gian, tách nền, log
    rag/               Chunking, embedding, kho Chroma, cache tóm tắt
    video/             Hỗ trợ WebVTT và thời gian
    youtube/           Lấy metadata/phụ đề qua yt-dlp và dự phòng Whisper
    hardware.py        Phát hiện RAM/VRAM và đề xuất cấu hình phần cứng
  workers/             Registry job nền gọn nhẹ
  tests/               Unit test (unittest, không cần GPU/tải model)

frontend/
  app/                 Route Next.js: thư viện, bản nháp, thêm video, chi tiết video
  components/          Player Vidstack và UI transcript
  lib/api.js           Client API frontend qua Next rewrites

docs/
  architecture.md      Ranh giới hệ thống và luồng dữ liệu
  dubbing.md           Chi tiết pipeline dubbing
  rag.md               Chi tiết index RAG và Q&A

scripts/
  measure_vram.py      Đo VRAM/RAM thực tế để calibrate các bảng tier phần cứng
```

## Tài liệu

Chi tiết triển khai nằm trong `docs/`:

- [Kiến trúc](docs/architecture.md)
- [Pipeline Dubbing](docs/dubbing.md)
- [Pipeline RAG](docs/rag.md)

README được giữ chủ đích làm điểm vào của dự án. Ghi chú thiết kế chi tiết và
hành vi pipeline nằm trong các file ở trên.

## Yêu cầu hệ thống

- GPU NVIDIA được khuyến nghị cho OmniVoice và chế độ Whisper GPU nhanh hơn.
  OmniVoice cần **≥ 3GB VRAM** (mức tối thiểu đo được ~2.5GB, +đệm cho
  driver/CUDA context overhead); dưới mức đó sẽ tự rơi về Supertonic CPU.
- Máy chỉ có CPU vẫn chạy được đường mặc định Supertonic + Whisper CPU, nhưng
  ASR và TTS sẽ chậm hơn.
- Chạy qua Docker Compose — xem "Cài đặt & Chạy (Docker)" bên dưới. Cần có
  [Docker Engine](https://docs.docker.com/engine/install/) kèm plugin Compose
  (Docker Desktop trên Windows/macOS).

## Cookie YouTube

Cookie là tuỳ chọn. Đa số video công khai vẫn tải bình thường không cần
cookie. Chỉ cần thêm cookie khi gặp 1 trong các trường hợp sau:

- Video giới hạn độ tuổi, riêng tư, hoặc chỉ dành cho thành viên.
- YouTube chặn request metadata/phụ đề/audio do phát hiện bot (hiếm gặp với
  dùng bình thường, dễ gặp hơn nếu request lặp lại nhiều/dồn dập).

**Trong setup Docker này, cookie bắt buộc phải là file** — app cũng hỗ trợ tự
phát hiện cookie từ trình duyệt cài sẵn trên máy (`YT_COOKIES_BROWSER=chrome`),
nhưng cách đó chỉ hoạt động khi app chạy trực tiếp trên máy có trình duyệt
thật. Container không có trình duyệt, nên `YT_COOKIES_BROWSER` đã bị tắt hẳn
trong `docker-compose.yml` và không có tác dụng gì ở đây — dùng cách export
file bên dưới thay thế.

```text
YT_COOKIES_DIR=cookies
# hoặc
YT_COOKIES_PATH=cookies/primary.txt
```

Cách tạo file cookie:

1. Cài extension trình duyệt export cookie theo định dạng Netscape
   `cookies.txt`, ví dụ "Get cookies.txt LOCALLY".
2. Đăng nhập YouTube trong trình duyệt đó.
3. Export cookie cho `youtube.com`.
4. Lưu file thành `cookies/primary.txt`.

Thư mục `cookies/` đã được git ignore. Không commit cookie đã export. File này
là thông tin đăng nhập — hãy đối xử như mật khẩu: cookie export có thể bị suy
giảm hoặc thu hồi nếu dùng theo kiểu bất thường, tần suất cao (tự động hoá),
nên tránh viết script gọi lặp lại liên tục vào nó.

## Web Search (tuỳ chọn)

RAG theo video vẫn hoạt động bình thường không cần web search. Stack Docker
(bên dưới) tự khởi động SearXNG cho bạn như một phần của toàn bộ stack; ghi đè
URL nếu bạn trỏ app sang một instance khác:

```text
SEARXNG_URL=http://localhost:8888
```

## Cài đặt & Chạy (Docker)

Cần [Docker Engine](https://docs.docker.com/engine/install/) kèm plugin
Compose (Docker Desktop trên Windows/macOS). Không cần cài Python, Node.js,
hay `ffmpeg` trên máy — mọi thứ chạy bên trong container.

### Biến môi trường

```bash
cp .env.example .env
```

**API key là tuỳ chọn, tuỳ vào tính năng bạn dùng:**

- Lồng tiếng ở **chế độ dịch thủ công** (copy prompt sang ChatGPT, dán kết quả
  vào): không cần API key nào cả.
- **Chế độ dịch qua API**: cần 1 key provider LLM (DeepSeek/OpenAI/
  Gemini/Anthropic).
- **Chat hỏi đáp RAG**: truy xuất và embedding chạy local miễn phí, nhưng câu
  trả lời và bản tóm tắt cache được sinh bởi LLM, nên cũng cần 1 key provider.

Chỉ cần điền provider/tính năng bạn dùng. Mặc định hiện tại:

- UI dịch qua API mặc định dùng DeepSeek Flash khi chọn chế độ API.
- UI RAG mặc định dùng DeepSeek trừ khi đổi trong `backend/config.yaml`.
- TTS mặc định dùng Supertonic (CPU).
- ASR mặc định dùng preset CPU trong `backend/config.yaml`.

Setup API tối thiểu cho DeepSeek:

```text
DEEPSEEK_API_KEY=...
```

Provider thay thế tuỳ chọn:

```text
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

### Chạy

Toàn bộ stack (backend + frontend + SearXNG) chạy được trong container:

```bash
docker compose up -d --build
```

- Frontend: `http://localhost:3000` — Backend: `http://localhost:8010` (chỉ loopback).
- Image mặc định là **CPU-only** (bản CPU của PyTorch, nhẹ hơn nhiều so với
  bản CUDA). ASR chạy `small.en` int8 và TTS dùng Supertonic.
- `./data` và `./cookies` được bind-mount ra máy host, nên media và phụ đề
  được sinh ra dễ kiểm tra từ bên ngoài container. Model tải về (Whisper,
  bge-m3, TTS) được cache trong volume `model-cache`, giữ nguyên qua các lần
  rebuild image.
- Lần chạy đầu tải vài GB model theo nhu cầu — các lần sau nhanh hơn nhiều.

**GPU (NVIDIA)** — bật preset Whisper CUDA và OmniVoice.

- **Linux**: cần cài
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  trên host trước (chỉ có driver là chưa đủ — đây là thứ giúp container nhìn
  thấy GPU).
- **Windows**: dùng Docker Desktop với backend WSL2 + driver NVIDIA mới —
  làm theo
  [hướng dẫn CUDA trên WSL2 của NVIDIA](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
  thay vì cài toolkit như trên.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Đã verify hoạt động đầu-cuối trên GPU RTX 4050 Laptop (host Fedora): container
nhận diện đúng GPU, `/api/hardware` đề xuất `gpu_turbo` + OmniVoice, image
~14GB (so với ~5.4GB bản CPU-only).

Ghi chú:

- Cookie: xem mục "Cookie YouTube" ở trên — thư mục `cookies/` đều được mount
  vào container ở cả 2 trường hợp.
- Trên host dùng SELinux (Fedora), file compose đã dùng cờ `:z` cho bind
  mount. File do container tạo trong `./data` sẽ thuộc quyền root trên host —
  chạy `sudo chown -R $USER data` nếu cần chỉnh sửa trực tiếp.
- Bản CPU và GPU dùng 2 tag image riêng (`tubenote-backend:cpu` và
  `tubenote-backend:gpu`), nên build bản này không ghi đè mất bản kia — có thể
  build và giữ cả 2, đổi qua lại chỉ cần chọn đúng lệnh `docker compose`
  tương ứng.

## Luồng sử dụng thông thường

1. Mở `Tạo lồng tiếng`.
2. Chọn model ASR. CPU là đường tương thích mặc định.
3. Nạp URL YouTube.
4. Chọn engine TTS trong panel TTS.
5. Chọn chế độ dịch:
   - Thủ công: copy prompt, dịch bên ngoài, dán kết quả vào.
   - API: chọn provider/model và để TubeNote dịch theo lô.
6. Kiểm tra các lô đã dịch.
7. Tuỳ chọn chỉnh giọng, chất lượng, và giữ nền.
8. Bắt đầu lồng tiếng.
9. Xem lại kết quả trong player.
10. Tạo lại từng đoạn khi cần sửa phát âm hoặc cách đọc.
11. Dùng tab Q&A để tạo/nạp bản tóm tắt và hỏi đáp về video.

## Dữ liệu runtime

Các file được sinh ra đều bị git ignore:

- `data/` (bao gồm cả giọng tham chiếu runtime trong `data/voice_clones/`)
- `cookies/`
- `.env`

`voice_clones/` ở gốc repository được commit có chủ đích: nó chứa các giọng
mẫu tích hợp sẵn cho 2 engine TTS, mà dropdown chọn giọng phụ thuộc vào.

Các đường dẫn runtime quan trọng:

```text
data/metadata/          JSON metadata YouTube
data/sub_raw/           Phụ đề thô từ YouTube hoặc Whisper
data/sub_vi_super/      JSON phụ đề/timing đã dịch, dùng Supertonic
data/sub_vi_omni/       JSON phụ đề/timing đã dịch, dùng OmniVoice
data/audio_dub/         Giọng nói đã lồng tiếng được sinh ra
data/video_dub/         File MP4 kết quả cuối cùng
data/chroma/            Kho vector RAG
data/rag_summary/       Bản tóm tắt video đã cache
data/logs/              Log CSV timing/hiệu năng
data/voice_clones/      Giọng tham chiếu runtime sinh ra từ video nguồn
```

## Giới hạn của dự án

- Trạng thái job runtime lưu trong bộ nhớ. Restart backend sẽ mất trạng thái
  job đang chạy.
- Trạng thái bản nháp lưu ở trình duyệt local trong lúc lồng tiếng, không phải
  database nhiều người dùng.
- Chất lượng TTS phụ thuộc vào thời gian phụ đề, độ dài bản dịch, hành vi
  model, và thời lượng từng đoạn.
- Việc giữ nền phụ thuộc vào nguồn; video nhiều nhạc thường ra kết quả tốt hơn
  video có tiếng nói ồn với giọng chồng lấn.
- Media được sinh ra và nội dung YouTube tải về được chủ đích loại khỏi
  repository công khai.

## License

MIT.
