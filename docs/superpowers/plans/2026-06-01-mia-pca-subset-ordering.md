# MIA PCA 기반 서브셋 정렬 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `make_nested_orders`의 랜덤 셔플을 PCA 중심 거리 기반 정렬로 교체해 서브셋 구성을 결정론적으로 만든다.

**Architecture:** pretrained ResNet34 avgpool(512-dim) feature → PCA fit(member+nonmember 합산) → centroid L2 거리 오름차순 정렬. `build_pca_order(member_files, nonmember_files)` 함수를 Split 섹션에 추가하고, `make_nested_orders` 내 shuffle 2줄을 이 함수 호출로 교체한다.

**Tech Stack:** PyTorch (ResNet34), sklearn.decomposition.PCA, numpy, pytest

---

## 파일 구조

| 파일 | 작업 |
|---|---|
| `d:\MJU\Extra\ED@IG\KADS\MIA.py` | `build_pca_order` 추가, `make_nested_orders` 수정, import 추가 |
| `d:\MJU\Extra\ED@IG\KADS\tests\test_mia_pca.py` | 신규 테스트 파일 |

---

### Task 1: 테스트 파일 작성 및 실패 확인

**Files:**
- Create: `d:\MJU\Extra\ED@IG\KADS\tests\test_mia_pca.py`

- [ ] **Step 1: 테스트 파일 생성**

`d:\MJU\Extra\ED@IG\KADS\tests\test_mia_pca.py`:

```python
import os
import sys
import random
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MIA import build_pca_order


def _make_images(n, directory):
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(42)
    files = []
    for i in range(n):
        arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        path = os.path.join(directory, f"img_{i}.jpg")
        Image.fromarray(arr).save(path)
        files.append(path)
    return files


def test_build_pca_order_returns_subsets(tmp_path):
    """반환 리스트가 입력의 부분집합인지 확인"""
    m = _make_images(10, str(tmp_path / "m"))
    nm = _make_images(10, str(tmp_path / "nm"))

    m_sorted, nm_sorted = build_pca_order(m, nm)

    assert len(m_sorted) <= len(m)
    assert len(nm_sorted) <= len(nm)
    assert set(m_sorted).issubset(set(m))
    assert set(nm_sorted).issubset(set(nm))


def test_build_pca_order_is_deterministic(tmp_path):
    """같은 입력 → 같은 출력 순서"""
    m = _make_images(10, str(tmp_path / "m"))
    nm = _make_images(10, str(tmp_path / "nm"))

    m1, nm1 = build_pca_order(m, nm)
    m2, nm2 = build_pca_order(m, nm)

    assert m1 == m2
    assert nm1 == nm2


def test_build_pca_order_input_order_independent(tmp_path):
    """입력 순서가 달라도 출력 순서는 동일해야 한다 (nested subset 보장)"""
    m = _make_images(10, str(tmp_path / "m"))
    nm = _make_images(10, str(tmp_path / "nm"))

    m_sorted, _ = build_pca_order(m, nm)

    shuffled = m[:]
    random.shuffle(shuffled)
    m_sorted2, _ = build_pca_order(shuffled, nm)

    assert m_sorted == m_sorted2
```

- [ ] **Step 2: 테스트 실패 확인**

```
cd d:\MJU\Extra\ED@IG\KADS
python -m pytest tests/test_mia_pca.py -v
```

Expected: `ImportError: cannot import name 'build_pca_order' from 'MIA'`

---

### Task 2: `build_pca_order` 구현

**Files:**
- Modify: `d:\MJU\Extra\ED@IG\KADS\MIA.py`

- [ ] **Step 1: import 추가**

`MIA.py` 상단 sklearn import 줄을 찾아 `PCA`를 추가한다:

```python
# 기존
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score

# 변경 후
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
```

- [ ] **Step 2: `build_pca_order` 함수 추가**

`split_ffhq` 함수(96줄) 아래, `make_nested_orders` 함수(105줄) 위에 삽입:

```python
def build_pca_order(member_files, nonmember_files, n_components=50):
    """
    member_files + nonmember_files로 PCA를 fit한 뒤,
    각 pool을 centroid까지 L2 거리 오름차순으로 정렬해 반환한다.
    가까울수록 전형적인(대표적인) 이미지.
    """
    extractor = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    extractor.fc = nn.Identity()
    extractor = extractor.to(DEVICE).eval()

    def _extract(files):
        ds = FaceDataset(files, label=0, transform=transform)
        loader = DataLoader(ds, batch_size=ATTACK_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        feats = []
        with torch.no_grad():
            for imgs, _ in loader:
                feats.append(extractor(imgs.to(DEVICE)).cpu().numpy())
        return np.vstack(feats), ds.files

    member_feats, valid_member = _extract(member_files)
    nonmember_feats, valid_nonmember = _extract(nonmember_files)

    combined = np.vstack([member_feats, nonmember_feats])
    pca = PCA(n_components=n_components)
    combined_proj = pca.fit_transform(combined)
    centroid = combined_proj.mean(axis=0)

    member_proj = pca.transform(member_feats)
    nonmember_proj = pca.transform(nonmember_feats)

    member_order = [valid_member[i] for i in np.argsort(np.linalg.norm(member_proj - centroid, axis=1))]
    nonmember_order = [valid_nonmember[i] for i in np.argsort(np.linalg.norm(nonmember_proj - centroid, axis=1))]

    return member_order, nonmember_order
```

> 주의: `FaceDataset`은 이 함수 아래 정의되어 있지만 호출 시점엔 이미 정의된 상태라 문제 없다.

- [ ] **Step 3: 테스트 통과 확인**

```
python -m pytest tests/test_mia_pca.py -v
```

Expected: 3개 PASS (첫 실행 시 ResNet34 가중치 다운로드로 시간 걸릴 수 있음)

- [ ] **Step 4: 커밋**

```
git add d:\MJU\Extra\ED@IG\KADS\MIA.py d:\MJU\Extra\ED@IG\KADS\tests\test_mia_pca.py
git commit -m "feat: add build_pca_order for deterministic subset selection"
```

---

### Task 3: `make_nested_orders` 수정

**Files:**
- Modify: `d:\MJU\Extra\ED@IG\KADS\MIA.py:105-124`

- [ ] **Step 1: `make_nested_orders` 교체**

현재 105~124줄을 아래로 교체:

```python
def make_nested_orders(real_files, fake_files, data_seed):
    """
    data_seed로 member_pool/nonmember_pool을 나누고,
    각 pool을 PCA 대표성 순으로 정렬한다.
    같은 data_seed면 n과 무관하게 앞쪽 n개가 항상 동일하다.
    """
    member_pool, nonmember_pool = split_ffhq(real_files, member_ratio=MEMBER_RATIO, seed=data_seed)

    print(f"  Building PCA order for data_seed={data_seed}...")
    member_order, nonmember_order = build_pca_order(member_pool, nonmember_pool)

    rng_fake = random.Random(data_seed + 303)
    fake_order = fake_files[:]
    rng_fake.shuffle(fake_order)

    return member_order, nonmember_order, fake_order
```

- [ ] **Step 2: 기존 테스트 재확인**

```
python -m pytest tests/test_mia_pca.py -v
```

Expected: 3개 PASS

- [ ] **Step 3: import 가능 확인**

```
python -c "from MIA import make_nested_orders; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: 커밋**

```
git add d:\MJU\Extra\ED@IG\KADS\MIA.py
git commit -m "feat: replace random shuffle with PCA ordering in make_nested_orders"
```
