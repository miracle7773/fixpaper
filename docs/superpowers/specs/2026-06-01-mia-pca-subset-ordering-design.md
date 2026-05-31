# MIA PCA 기반 서브셋 정렬 설계

**날짜:** 2026-06-01  
**파일:** `d:\MJU\Extra\ED@IG\KADS\MIA.py`

## 문제

현재 `make_nested_orders`에서 member_pool, nonmember_pool을 랜덤 셔플 후 앞에서 n개 자른다. data_seed마다 어떤 이미지가 각 서브셋에 들어갈지 불확실하고, 서브셋 구성의 임의성이 n 효과와 혼재된다.

## 목표

서브셋 선택 기준을 랜덤에서 PCA 기반 대표성 순으로 교체한다. n=250은 항상 가장 전형적인 250장, n=750은 가장 전형적인 750장이 된다. data_seed는 여전히 member_pool / nonmember_pool 배정(split_ffhq)에만 영향을 주고, pool 안에서의 순서는 PCA로 고정된다.

## 설계

### 새 함수: `build_pca_order(member_files, nonmember_files, n_components=50)`

**위치:** Split 섹션, `split_ffhq` 아래, `make_nested_orders` 위

**동작:**
1. member_files + nonmember_files 합쳐서 ImageNet pretrained ResNet34로 512-dim feature 추출
2. 합쳐진 feature로 PCA fit → 공통 PCA 공간 확보
3. member_files, nonmember_files 각각 PCA 공간에 투영 후 centroid까지 L2 거리 계산
4. 각각 거리 오름차순 정렬 → 중심에 가까운(전형적인) 이미지가 앞으로
5. (member_sorted, nonmember_sorted) 튜플 반환

**파라미터:**
- `n_components=50`: 512차원 중 주요 분산 커버, 변경 가능
- feature extractor: 학습 모델과 별개인 pretrained ResNet34만 사용
- PCA는 두 pool 합쳐서 한 번만 fit → 동일한 기준으로 두 pool 비교 보장

### 변경: `make_nested_orders`

**현재:**
```
rng_member.shuffle(member_order)
rng_nonmember.shuffle(nonmember_order)
rng_fake.shuffle(fake_order)
```

**변경 후:**
```
member_pool + nonmember_pool 합쳐서 build_pca_order로 PCA fit
member_order    → PCA 중심 거리 순 정렬
nonmember_order → 동일 PCA 공간에서 중심 거리 순 정렬
fake_order      → 랜덤 셔플 유지 (MIA 평가 대상 아님)
```

두 pool을 합쳐서 PCA를 fit하는 이유: 같은 PCA 공간에서 member/non-member를 동일한 기준으로 비교해야 분포 mismatch가 없다.

### 변경 없는 부분

- `split_ffhq`: data_seed 기반 pool 배정 그대로
- `SUBSET_SIZES = [250, 750, 1250, 2500]`: 그대로
- nested subset 구조 (앞에서 n개 자르기): 그대로
- `split_files_train_val_nested`, `split_attack_files`: 그대로
- 이후 모든 학습/평가 코드: 변경 없음

## 결과 해석

- AUC, TPR 등 MIA 지표 해석 방향 변화 없음
- data_seed 간 variance 감소 예상 (seed가 ordering에 영향 못 줌)
- n 효과(서브셋 크기 증가 → 과적합 감소 → MIA 신호 약화)는 그대로 유지

## 변경 범위

- 추가: `build_pca_order` 함수 1개
- 수정: `make_nested_orders` 내 shuffle 2줄 → `build_pca_order` 호출로 교체
- 나머지 코드: 손 안 댐
