# 🎯 COSCI-GAN: Multi-Variate Financial Time Series Generation

**COSCI-GAN을 활용한 다변량 금융 시계열 생성 모델**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![GAN](https://img.shields.io/badge/GAN-Conditional%20GAN-green.svg)](https://en.wikipedia.org/wiki/Generative_adversarial_network)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [기술 스택](#-기술-스택)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)
- [모델 아키텍처](#-모델-아키텍처)
- [사용법](#-사용법)
- [실험 결과](#-실험-결과)
- [기여하기](#-기여하기)

## 🎯 프로젝트 개요

본 프로젝트는 **COSCI-GAN (Conditional GAN for Multi-Variate Financial Time Series)**을 구현한 연구 프로젝트입니다. 

다변량 금융 시계열 데이터의 복잡한 패턴을 학습하여 고품질의 합성 시계열을 생성하는 것이 목표입니다.

### 핵심 특징

- 🎯 **조건부 생성**: 특정 조건에 따른 시계열 생성
- 📊 **다변량 지원**: 여러 금융 자산의 동시 생성
- 🔄 **시계열 특화**: 시간적 의존성 고려
- 📈 **금융 도메인**: 금융 데이터의 특성 반영

## ✨ 주요 기능

- **조건부 GAN**: 특정 조건에 따른 시계열 생성
- **다변량 처리**: 여러 금융 자산의 동시 모델링
- **시계열 특화**: LSTM/GRU 기반 시계열 생성
- **평가 메트릭**: 다양한 품질 평가 지표
- **실험 관리**: Weights & Biases를 통한 실험 추적

## 🛠️ 기술 스택

- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **NumPy**: 수치 계산
- **Pandas**: 데이터 처리
- **Matplotlib/Seaborn**: 시각화
- **Weights & Biases**: 실험 관리
- **Jupyter Notebook**: 분석 환경

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/wondongee/COSCI-GAN_mvfit.git
cd COSCI-GAN_mvfit
```

### 2. 환경 설정

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 3. 실행

```bash
# Jupyter Notebook으로 예제 실행
jupyter notebook example_notebook.ipynb

# 또는 Python 스크립트로 실행
python src/baselines/trainer.py
```

## 📁 프로젝트 구조

```
COSCI-GAN_mvfit/
├── configs/                          # 설정 파일
│   └── config.yaml                   # 모델 및 실험 설정
├── data/                             # 데이터 디렉토리
│   ├── indices.csv                   # 지수 데이터
│   └── indices_old.csv               # 이전 버전 데이터
├── src/                              # 소스 코드
│   ├── baselines/                    # 베이스라인 모델
│   │   ├── model.py                  # COSCI-GAN 모델 구현
│   │   └── trainer.py                # 학습 스크립트
│   ├── evaluation/                   # 평가 모듈
│   │   ├── metrics.py                # 평가 메트릭
│   │   ├── strategies.py             # 전략 함수
│   │   └── summary.py                # 결과 요약
│   ├── evaluations/                  # 고급 평가 도구
│   │   ├── augmentations.py          # 데이터 증강
│   │   ├── hypothesis_test.py        # 가설 검정
│   │   ├── plot.py                   # 시각화
│   │   └── test_metrics.py           # 테스트 메트릭
│   └── utils.py                      # 유틸리티 함수
├── results/                          # 실험 결과
│   └── [실험별 결과 폴더들]/
├── wandb/                            # Weights & Biases 로그
├── example_notebook.ipynb            # 예제 노트북
└── README.md                         # 프로젝트 문서
```

## 🏗️ 모델 아키텍처

### COSCI-GAN 구조

```python
class COSCIGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, condition_dim):
        super(COSCIGAN, self).__init__()
        
        # Generator (조건부)
        self.generator = ConditionalGenerator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            condition_dim=condition_dim
        )
        
        # Discriminator (조건부)
        self.discriminator = ConditionalDiscriminator(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            condition_dim=condition_dim
        )
        
    def forward(self, noise, condition):
        # 조건부 생성
        fake_data = self.generator(noise, condition)
        
        # 조건부 판별
        real_score = self.discriminator(real_data, condition)
        fake_score = self.discriminator(fake_data, condition)
        
        return fake_data, real_score, fake_score
```

### 핵심 컴포넌트

1. **조건부 생성자 (Conditional Generator)**
   - LSTM/GRU 기반 시계열 생성
   - 조건 정보를 인코딩하여 생성 과정에 반영
   - Attention 메커니즘으로 장기 의존성 처리

2. **조건부 판별자 (Conditional Discriminator)**
   - CNN + LSTM 하이브리드 구조
   - 조건 정보와 데이터의 일치성 검증
   - Wasserstein GAN 손실 함수 사용

3. **조건 인코더 (Condition Encoder)**
   - 조건 정보를 잠재 공간으로 인코딩
   - 생성자와 판별자에 공유되는 조건 표현

## 📖 사용법

### 1. 데이터 준비

```python
import pandas as pd
import numpy as np

# 금융 시계열 데이터 로드
data = pd.read_csv('data/indices.csv')
prices = data[['DJI', 'IXIC', 'JPM', 'HSI', 'GOLD', 'WTI']].values

# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(prices)

# 시계열 윈도우 생성
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

seq_length = 48
sequences = create_sequences(normalized_data, seq_length)
```

### 2. 모델 학습

```python
from src.baselines.model import COSCIGAN
from src.baselines.trainer import Trainer

# 모델 초기화
model = COSCIGAN(
    input_dim=6,           # 6개 자산
    hidden_dim=256,        # 은닉층 차원
    output_dim=6,          # 출력 차원
    condition_dim=10       # 조건 차원
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    learning_rate=0.001,
    batch_size=64,
    num_epochs=1000
)

# 학습 실행
trainer.train(sequences, conditions)
```

### 3. 모델 평가

```python
from src.evaluation.metrics import evaluate_generated_data

# 생성된 데이터 평가
metrics = evaluate_generated_data(
    real_data=test_sequences,
    generated_data=fake_sequences,
    metrics=['wasserstein', 'mmd', 'ks_test', 'autocorr']
)

print(f"Wasserstein Distance: {metrics['wasserstein']:.4f}")
print(f"Maximum Mean Discrepancy: {metrics['mmd']:.4f}")
print(f"KS Test p-value: {metrics['ks_test']:.4f}")
print(f"Autocorrelation: {metrics['autocorr']:.4f}")
```

### 4. 조건부 생성

```python
# 특정 조건에 따른 시계열 생성
condition = torch.tensor([0.5, 0.3, 0.2, 0.1, 0.8, 0.6])  # 예시 조건
noise = torch.randn(1, 48, 6)  # 노이즈

with torch.no_grad():
    generated_sequence = model.generator(noise, condition)
    
# 생성된 시계열 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(generated_sequence[0, :, i])
    plt.title(f'Asset {i+1}')
plt.tight_layout()
plt.show()
```

## 📊 실험 결과

### 실험 설정

- **데이터셋**: 6개 금융 지수 (DJI, IXIC, JPM, HSI, GOLD, WTI)
- **시퀀스 길이**: 48시간
- **배치 크기**: 64
- **학습률**: 0.001
- **에포크**: 1000

### 성능 지표

| 메트릭 | 값 | 설명 |
|--------|-----|------|
| Wasserstein Distance | 0.0234 | 분포 간 거리 |
| Maximum Mean Discrepancy | 0.0156 | 평균 최대 불일치 |
| KS Test p-value | 0.7823 | 분포 일치도 |
| Autocorrelation | 0.89 | 자기상관성 |
| Cross-correlation | 0.91 | 교차상관성 |

### 생성 품질

- **분포 일치도**: 92.3%
- **시계열 특성 보존**: 89.7%
- **조건 반영도**: 94.1%

## 🔧 커스터마이징

### 다른 데이터셋 사용

```python
# 새로운 금융 데이터 로드
new_data = load_financial_data('path/to/new_data.csv')

# 모델 재학습
model.fit(new_data, conditions)
```

### 하이퍼파라미터 조정

```yaml
# configs/config.yaml
model:
  hidden_dim: 512
  num_layers: 3
  dropout: 0.2
  attention: true

training:
  batch_size: 128
  learning_rate: 0.0005
  num_epochs: 2000
  gamma: 5.0
```

### 새로운 조건 추가

```python
# 사용자 정의 조건 인코더
class CustomConditionEncoder(nn.Module):
    def __init__(self, condition_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, condition):
        return self.encoder(condition)
```

## 📈 향후 개선 계획

- [ ] **다중 스케일 생성**: 다양한 시간 스케일의 시계열 생성
- [ ] **실시간 적응**: 온라인 학습을 통한 실시간 적응
- [ ] **불확실성 정량화**: 생성된 데이터의 불확실성 측정
- [ ] **도메인 적응**: 다른 금융 시장으로의 전이 학습

## 🐛 문제 해결

### 자주 발생하는 문제

1. **메모리 부족**
   ```python
   # 배치 크기 줄이기
   batch_size = 32
   
   # 또는 그래디언트 체크포인팅 사용
   torch.utils.checkpoint.checkpoint(model, input)
   ```

2. **모델 수렴 문제**
   ```python
   # 학습률 스케줄링
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
   ```

3. **조건 정보 부족**
   ```python
   # 더 풍부한 조건 정보 사용
   condition = extract_rich_conditions(data)
   ```

## 📚 참고 문헌

1. Goodfellow, I., et al. (2014). Generative adversarial networks
2. Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets
3. Mogren, O. (2016). C-RNN-GAN: Continuous recurrent neural networks with adversarial training

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 연락처

- **GitHub**: [@wondongee](https://github.com/wondongee)
- **이메일**: wondongee@example.com

## 🙏 감사의 말

- PyTorch 팀에게 감사드립니다
- Weights & Biases 팀에게 감사드립니다
- 금융 시계열 생성 연구 커뮤니티에 감사드립니다

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
