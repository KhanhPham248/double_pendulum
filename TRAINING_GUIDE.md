# Training Guide

Tài liệu này mô tả quy trình chuyển project sang máy có GPU, kiểm tra môi
trường và bắt đầu huấn luyện task `combined` bằng SAC hoặc PPO.

## 1. Chuyển project sang máy train

Có thể clone repository bằng Git. Nếu chuyển trực tiếp từ máy dev, dùng:

```bash
rsync -av \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'runs/' \
  /home/khanh248/Documents/HB/Mujoco/double_pendulum/ \
  USER@TRAIN_MACHINE:/PATH/TO/double_pendulum/
```

Thư mục `runs/` được bỏ qua vì có thể chứa checkpoint lớn. Xóa dòng
`--exclude 'runs/'` nếu cần chuyển một run cũ để tiếp tục train.

## 2. Yêu cầu máy train

- Linux x86-64;
- Python 3.10 đến 3.13, khuyến nghị Python 3.12;
- NVIDIA GPU và driver tương thích với bản PyTorch CUDA;
- đủ RAM, VRAM và dung lượng đĩa cho replay buffer và checkpoint.

Không bắt buộc cài CUDA Toolkit hệ thống nếu PyTorch wheel đã chứa CUDA
runtime, nhưng NVIDIA driver vẫn phải tương thích.

## 3. Cài môi trường

### Sử dụng `uv`

```bash
cd /PATH/TO/double_pendulum
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
```

### Sử dụng `venv` và `pip`

```bash
cd /PATH/TO/double_pendulum
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Project dùng `rsl-rl-lib==5.0.1` để tương thích với phần mJLab được đóng gói
kèm source. Không nâng riêng RSL-RL nếu chưa kiểm tra lại adapter và test.

## 4. Kiểm tra trước khi train

```bash
nvidia-smi
python scripts/preflight.py
pytest -q
```

Đăng nhập W&B một lần trên máy train nếu muốn đồng bộ dashboard:

```bash
wandb login
```

Chỉ bắt đầu train GPU khi preflight kết thúc bằng:

```text
PREFLIGHT=PASS
```

Test pass xác nhận dependency, wiring và policy contract; đây chưa phải bằng
chứng policy có thể hội tụ.

Có thể kiểm tra riêng model vật lý bằng MuJoCo Viewer:

```bash
python -m mujoco.viewer --mjcf assets/double_pendulum.xml
```

## 5. Train baseline

Nên bắt đầu với `64` environment để đo VRAM và throughput, sau đó mới tăng lên
`128`, `256` hoặc `512`.

### SAC

```bash
python scripts/train_sac.py \
  --task combined \
  --device cuda:0 \
  --num-envs 64 \
  --total-transitions 2000000 \
  --batch-size 256 \
  --utd-ratio 0.25 \
  --evaluation-episodes 10 \
  --wandb \
  --wandb-project double-pendulum \
  --wandb-run-name sac-v4-seed1 \
  --export-fail-fast
```

Reward-v4 SAC dùng checkpoint format `double_pendulum_sac_v3`, trong đó có
trạng thái update budget và reward metadata. Không resume checkpoint SAC cũ
được tạo với reward-v1/v2/v3; hãy bắt đầu một run mới.

Run kiểm chứng reward-v4 đầu tiên:

```bash
python scripts/train_sac.py \
  --task combined \
  --device cuda:0 \
  --num-envs 64 \
  --total-transitions 2000000 \
  --batch-size 256 \
  --utd-ratio 0.25 \
  --seed 1 \
  --run-dir runs/combined/sac/reward_v4_seed1 \
  --export-fail-fast
```

### PPO với mJLab/RSL-RL

```bash
python scripts/train_ppo.py \
  --task combined \
  --device cuda:0 \
  --num-envs 64 \
  --max-iterations 2000 \
  --evaluation-episodes 10 \
  --wandb \
  --wandb-project double-pendulum \
  --wandb-run-name ppo-v4-seed1 \
  --export-fail-fast
```

Task `combined` hiện dùng reward formula v4. Formula này giữ mục tiêu upright
của v3 (`q1 = pi`, `q2 = 0`) và thêm penalty vận tốc toàn cục để tránh nghiệm
quay vòng tốc độ cao. Reward version và toàn bộ weight được lưu trong
`task_config.yaml` và `policy.yaml` của run mới.

### PPO reward-v4 stability run

Cấu hình mặc định mới dùng learning rate cố định `3e-4`. Cấu hình này tránh
learning rate adaptive tăng đến `1e-2`, nguyên nhân đi kèm value-loss explosion
và policy collapse trong run reward-v4 cũ:

RSL-RL kiểm tra actor KL sau từng mini-batch và nhân hoặc chia learning rate
cho `1.5`. Với 5 epoch và 4 mini-batch, scheduler có thể đổi learning rate 20
lần trong một iteration. Actor và critic dùng chung optimizer, nên learning
rate của critic cũng tăng dù scheduler chỉ đo actor KL. Khi actor gần hội tụ,
KL thấp làm learning rate tăng sai thời điểm và critic mất ổn định.

```bash
python scripts/train_ppo.py \
  --task combined \
  --device cuda:0 \
  --num-envs 512 \
  --max-iterations 2000 \
  --save-interval 50 \
  --evaluation-episodes 10 \
  --seed 1 \
  --run-dir runs/combined/ppo/reward_v4_stable_seed1 \
  --wandb \
  --wandb-project double-pendulum \
  --wandb-run-name ppo-v4-stable-seed1 \
  --export-fail-fast
```

Không resume checkpoint reward-v1/v2/v3 cho thí nghiệm này. Policy phải học
lại từ đầu để kết quả A/B có ý nghĩa.

Sau khi train, kiểm tra từ trạng thái upright trước:

```bash
python scripts/evaluate.py \
  --run runs/combined/ppo/reward_v4_seed1 \
  --episodes 100 \
  --duration 60 \
  --reset-mode upright
```

Sau đó kiểm tra toàn bộ swing-up từ trạng thái hanging:

```bash
python scripts/evaluate.py \
  --run runs/combined/ppo/reward_v4_seed1 \
  --episodes 100 \
  --duration 20 \
  --reset-mode hanging
```

Các dấu hiệu reward fix hoạt động đúng:

- action tại upright không còn liên tục chạm `-1` hoặc `+1`;
- `action_saturation_fraction` giảm rõ rệt so với baseline `96.37%`;
- `longest_hold_s` vượt 5 giây từ upright;
- reward `quiet_upright` và `upright_capture` tăng trong TensorBoard;
- reward `global_velocity` không bị âm lớn kéo dài;
- reward không còn bị một thành phần giữ khuỷu thẳng chi phối.

Xem toàn bộ tham số được hỗ trợ:

```bash
python scripts/train_sac.py --help
python scripts/train_ppo.py --help
```

## 6. Output của một run

```text
runs/<task>/<algorithm>/<timestamp>/
├── checkpoints/
│   ├── step_XXXXXXXXXXXX.pt hoặc model_*.pt
│   └── latest.pt
├── policy.onnx
├── policy.yaml
├── best/
│   ├── policy.onnx
│   ├── policy.yaml
│   ├── evaluation.json
│   └── checkpoint.txt
├── evaluations/
│   └── step_XXXXXXXXXXXX.json
├── metrics.jsonl hoặc TensorBoard logs
└── videos/
```

Checkpoint native được giữ để resume. Trong mỗi run chỉ có một
`policy.onnx`; file này được cập nhật bằng actor deterministic mới nhất sau khi
export và parity check thành công. Nếu export lỗi, checkpoint vẫn được giữ và
ONNX hợp lệ trước đó không bị thay thế.

Mỗi lần lưu checkpoint, trainer chạy deterministic evaluation từ trạng thái
`hanging`. Các metric `eval/hanging/*` được ghi vào TensorBoard và W&B. Thư mục
`best/` chỉ cập nhật khi policy mới tốt hơn theo thứ tự: success rate, thời gian
giữ lâu nhất, upright fraction, episode return và torque.

Theo dõi log bằng:

```bash
tensorboard --logdir runs --port 6006
```

Nếu máy train ở xa, tạo SSH tunnel từ máy local:

```bash
ssh -L 6006:localhost:6006 USER@TRAIN_MACHINE
```

Sau đó mở `http://localhost:6006`.

Khi chạy với `--wandb`, log TensorBoard local vẫn được giữ và được đồng bộ lên
W&B. Link của run được in ở dòng `WANDB_RUN=...`. Các biểu đồ quan trọng để
nhận biết bão hòa hoặc collapse là:

- `eval/hanging/success_rate` và `eval/hanging/longest_hold_s`;
- `eval/hanging/action_saturation_fraction`;
- PPO: `Loss/value`, `Policy/mean_std`, `Loss/learning_rate`;
- SAC: `sac/critic_loss`, `sac/actor_grad_norm`, `sac/learning_rate`.

Đừng chọn checkpoint chỉ dựa trên `train/stable_fraction`. Metric đó là snapshot
của vector environment; `eval/hanging/success_rate` mới kiểm tra trọn vẹn quá
trình swing-up bằng actor deterministic.

## 7. Resume training

### SAC

```bash
python scripts/train_sac.py \
  --task combined \
  --device cuda:0 \
  --num-envs 64 \
  --total-transitions 4000000 \
  --resume runs/combined/sac/<RUN>/checkpoints/latest.pt \
  --export-fail-fast
```

### PPO

```bash
python scripts/train_ppo.py \
  --task combined \
  --device cuda:0 \
  --num-envs 64 \
  --max-iterations 1000 \
  --resume runs/combined/ppo/<RUN>/checkpoints/latest.pt \
  --export-fail-fast
```

Không truyền `--run-dir` cùng lúc với `--resume`. Runner tự tiếp tục ghi vào
run chứa checkpoint.

## 8. Đánh giá policy

Đánh giá khả năng swing-up từ trạng thái treo:

```bash
python scripts/evaluate.py \
  --run runs/combined/sac/<RUN> \
  --episodes 100 \
  --duration 20 \
  --reset-mode hanging
```

Đánh giá balancing dài hạn từ trạng thái dựng đứng:

```bash
python scripts/evaluate.py \
  --run runs/combined/ppo/<RUN> \
  --episodes 100 \
  --duration 60 \
  --reset-mode upright
```

Kiểm tra trực quan bằng Python MuJoCo runtime:

```bash
python scripts/simulate.py \
  --run runs/combined/sac/<RUN> \
  --duration 20 \
  --reset-mode hanging
```

Runtime dùng `policy.onnx` và `policy.yaml`, không phụ thuộc Torch, mJLab hoặc
RSL-RL. Khi chuyển riêng policy sang một máy mô phỏng khác, cần copy:

```text
assets/double_pendulum.xml
runs/<task>/<algorithm>/<RUN>/policy.onnx
runs/<task>/<algorithm>/<RUN>/policy.yaml
```

## 9. Tiêu chí kiểm tra ban đầu

- Train không xuất hiện NaN/Inf hoặc lỗi CUDA.
- `policy.onnx` được cập nhật sau checkpoint và vượt qua parity check.
- Swing-up được đánh giá trên nhiều episode và nhiều seed.
- Policy giữ vùng cân bằng liên tục ít nhất 5 giây trong baseline 20 giây.
- Balancing dài hạn được kiểm tra riêng trong episode 60 giây hoặc lâu hơn.
- Kiểm tra thêm perturbation, sai lệch tham số vật lý và action saturation.

Không xem import pass, unit test pass hoặc một gradient update là bằng chứng
policy đã học thành công. Acceptance cuối cùng phải dựa trên hành vi closed-loop
trong MuJoCo và kết quả evaluation nhiều episode.
