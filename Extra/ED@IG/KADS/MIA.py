import os
import random
import tempfile
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
# Configuration : 설정값
# REAL_PATH, FAKE_PATH 수정 필요
REAL_PATH = r"D:\kagglehub\datasets\xhlulu\140k-real-and-fake-faces\versions\2\real_vs_fake\real-vs-fake\train\real"
FAKE_PATH = r"D:\kagglehub\datasets\xhlulu\140k-real-and-fake-faces\versions\2\real_vs_fake\real-vs-fake\train\fake"

MEMBER_RATIO = 0.5  # member/non-member 비율

# [수정 위치 1] 기존 SUBSET_RATIOS 대신 실제 n 기준 실험으로 변경
# 같은 data seed 안에서는 nested subset 사용: n=250 ⊂ n=750 ⊂ n=1250 ⊂ n=2500
SUBSET_SIZES = [250, 750, 1250, 2500]

BATCH_SIZE = 8
ATTACK_BATCH_SIZE = 16
N_AUG = 5
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 20  # 최대 epoch 횟수 (추후 early stopping 가능)

# [수정 위치 2] data split seed와 model run seed를 분리
# DATA_SEEDS: member_pool/nonmember_pool split 및 nested subset 자체가 바뀌는 상위 반복
# NUM_MODEL_RUNS: 같은 data seed/subset에서 모델 초기화, train/val split, batch 순서만 바뀌는 반복
DATA_SEEDS = [1, 2, 3]
NUM_MODEL_RUNS = 3

# [수정 위치 3] train/validation split과 MIA attack validation/test split 비율 명시
TRAIN_VAL_RATIO = 0.9
ATTACK_VAL_RATIO = 0.5

# [수정 위치 4] checkpoint 저장 폴더 분리
CHECKPOINT_DIR = Path("checkpoints_strict_nested_xgb_report_v1")
USE_SAVED_MODELS = False

# ── Reproducibility ───────────────────────────────────────────────────────────
# Reproducibility : random seed 고정
def set_global_seed(seed):
    """random, numpy, torch seed를 한 번에 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(SEED)

def _collect(results, key):
    return np.array([r[key] for r in results if r.get(key) is not None], dtype=float)

def _mean(results, key):
    vals = _collect(results, key)
    return float(np.mean(vals)) if len(vals) else None

# ── Transform ─────────────────────────────────────────────────────────────────
# Transform: 이미지 size 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

aug_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Split ─────────────────────────────────────────────────────────────────────
# Split: real 중 member/non-member 분할
def split_ffhq(files, member_ratio=0.5, seed=42):
    """FFHQ 전체를 무작위로 섞어 member_pool과 nonmember_pool(holdout)로 분리."""
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    n_member = int(len(files) * member_ratio)
    return files[:n_member], files[n_member:]

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
        if len(ds.files) < len(files):
            warnings.warn(f"build_pca_order: {len(files) - len(ds.files)} files dropped by FaceDataset")
        return np.vstack(feats), ds.files

    member_feats, valid_member = _extract(member_files)
    nonmember_feats, valid_nonmember = _extract(nonmember_files)

    combined = np.vstack([member_feats, nonmember_feats])
    n_components = min(n_components, combined.shape[0])
    pca = PCA(n_components=n_components)
    combined_proj = pca.fit_transform(combined)
    centroid = combined_proj.mean(axis=0)

    n_member = len(member_feats)
    member_proj = combined_proj[:n_member]
    nonmember_proj = combined_proj[n_member:]

    member_order = [valid_member[i] for i in np.argsort(np.linalg.norm(member_proj - centroid, axis=1))]
    nonmember_order = [valid_nonmember[i] for i in np.argsort(np.linalg.norm(nonmember_proj - centroid, axis=1))]

    return member_order, nonmember_order

# [수정 위치 5] nested subset 생성을 위해 data seed별 고정 순서를 만든다.
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

# [수정 위치 6] train/validation split을 file list 단계에서 수행해 member 정의를 엄격히 관리
def split_files_train_val(files, train_ratio, seed):
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)

    if len(files) < 2:
        raise ValueError("Need at least 2 files for train/validation split.")

    n_train = int(len(files) * train_ratio)
    n_train = max(1, min(n_train, len(files) - 1))
    return files[:n_train], files[n_train:]


def split_files_train_val_nested(files, train_ratio):
    """Shuffle 없이 앞쪽 슬라이스로 고정 — nested subset 성질과 run 간 member 일관성을 보장한다."""
    if len(files) < 2:
        raise ValueError("Need at least 2 files for train/validation split.")
    n_train = int(len(files) * train_ratio)
    n_train = max(1, min(n_train, len(files) - 1))
    return files[:n_train], files[n_train:]

# [수정 위치 7] threshold 선택용 attack validation과 최종 평가용 attack test 분리
def split_attack_files(member_train_files, nonmember_files, attack_val_ratio, seed):
    """
    MIA member는 실제 train에 들어간 real만 사용한다.
    validation real은 MIA 평가에서 제외한다.
    """
    n_eval = min(len(member_train_files), len(nonmember_files))
    if n_eval < 2:
        raise ValueError("Need at least 2 member/nonmember files for attack split.")

    rng = random.Random(seed)
    members = rng.sample(member_train_files, n_eval)
    nonmembers = rng.sample(nonmember_files, n_eval)

    n_val = int(n_eval * attack_val_ratio)
    n_val = max(1, min(n_val, n_eval - 1))

    return {
        "member_attack_val": members[:n_val],
        "member_attack_test": members[n_val:],
        "nonmember_attack_val": nonmembers[:n_val],
        "nonmember_attack_test": nonmembers[n_val:],
    }

# ── Dataset ───────────────────────────────────────────────────────────────────
# Dataset: 이미지 전처리(Data Cleaning, 기타)
class FaceDataset(Dataset):
    def __init__(self, files, label, transform=None):
        self.label = label
        self.transform = transform
        valid = []
        for fpath in files:
            if os.path.getsize(fpath) == 0:
                continue
            try:
                with Image.open(fpath) as img:
                    img.verify()
                valid.append(fpath)
            except Exception:
                pass
        self.files = valid

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label

# ── Model ─────────────────────────────────────────────────────────────────────
# Model: ResNet34 불러오기
def build_model():
    try:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.resnet34(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2)

    # Freeze BatchNorm statistics (작은 subset에서 BN noise 방지)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    return model.to(DEVICE)

# ── Training ──────────────────────────────────────────────────────────────────
# Training: ResNet34 학습 (real/fake 분류기 학습)
"""
Loss: CrossEntropy
Optimizer: Adam
Learning Rate: CosineAnnealingLR

Early Stopping -> Validation Loss 기준 판단(학습 중단)
Best Checkpoint 저장 -> Validation Loss 최저점 저장 후 복원
"""
def train_model(model, train_loader, epochs, val_loader, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    patience = 5
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    no_improve = 0
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    best_ckpt_path = tmp.name
    tmp.close()

    for epoch in range(epochs):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
        val_loss /= max(1, len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = avg_train_loss
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            acc = 100.0 * correct / max(1, total)
            print(f"      Epoch [{epoch+1:3d}/{epochs}]  loss={avg_train_loss:.4f}  acc={acc:.1f}%  val_loss={val_loss:.4f}")

        if no_improve >= patience:
            print(f"      Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    os.remove(best_ckpt_path)
    return best_val_loss, best_train_loss

# ── Per-image Signals ─────────────────────────────────────────────────────────
# Per-image Signals: loss, confidence, entropy 신호 계산(수치화)
def compute_signals(model, dataset, aug_seed=0):
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    loader = DataLoader(dataset, batch_size=ATTACK_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    losses = []
    confidences = []
    entropies = []
    margins = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            probs = torch.softmax(out, dim=1)
            # true-class confidence (MIA 문헌 기준)
            conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            entr = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            true_logits = out.gather(1, labels.unsqueeze(1)).squeeze(1)
            other_logits = out.clone()
            other_logits.scatter_(1, labels.unsqueeze(1), float("-inf"))
            max_other_logits = other_logits.max(dim=1).values
            margin = true_logits - max_other_logits
            losses.extend(loss.cpu().numpy().tolist())
            confidences.extend(conf.cpu().numpy().tolist())
            entropies.extend(entr.cpu().numpy().tolist())
            margins.extend(margin.cpu().numpy().tolist())

    aug_ds = FaceDataset(dataset.files, dataset.label, transform=aug_transform)
    aug_loader = DataLoader(aug_ds, batch_size=ATTACK_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    torch.manual_seed(aug_seed)
    random.seed(aug_seed)
    aug_loss_runs = []
    with torch.no_grad():
        for _ in range(N_AUG):
            run_losses = []
            for imgs, labels in aug_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                run_losses.extend(loss.cpu().numpy().tolist())
            aug_loss_runs.append(run_losses)
    aug_arr = np.array(aug_loss_runs)
    return {
        "losses": np.array(losses),
        "confidence": np.array(confidences),
        "entropy": np.array(entropies),
        "margin": np.array(margins),
        "aug_loss_mean": aug_arr.mean(axis=0),
        "aug_loss_std":  aug_arr.std(axis=0),
    }


# ── MIA Evaluation ────────────────────────────────────────────────────────────
# MIA Evaluation: real[member] 와 real[non-member] 구분
def _safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def standardized_gap(member_scores, nonmember_scores):
    """pooled-std로 정규화한 separation d를 반환한다."""
    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    gap = float(np.mean(member_scores) - np.mean(nonmember_scores))
    if len(member_scores) < 2 or len(nonmember_scores) < 2:
        return gap, np.nan
    pooled = np.sqrt((np.var(member_scores, ddof=1) + np.var(nonmember_scores, ddof=1)) / 2.0)
    d_value = gap / pooled if pooled > 0 else 0.0
    return gap, float(d_value)


def choose_threshold_on_attack_val(member_signals, nonmember_signals):
    """
    [수정 위치 8] threshold 기반 precision/recall의 낙관적 bias를 줄이기 위해
    attack validation set에서만 threshold를 선택한다.
    """
    labels = np.concatenate([
        np.ones(len(member_signals["losses"])),
        np.zeros(len(nonmember_signals["losses"])),
    ])
    scores = np.concatenate([
        -member_signals["losses"],
        -nonmember_signals["losses"],
    ])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[j_idx])

def evaluate_mia(member_signals, nonmember_signals, threshold):
    labels = np.concatenate([
        np.ones(len(member_signals["losses"])),  # real[member] -> 1
        np.zeros(len(nonmember_signals["losses"])),  # real[non-member] -> 0
    ])

    def _run_auc(score_member, score_nonmember):
        scores = np.concatenate([score_member, score_nonmember])
        return _safe_auc(labels, scores)

    loss_scores_m = -member_signals["losses"]
    loss_scores_nm = -nonmember_signals["losses"]

    # 평가 지표 계산
    """
    AUC loss/confidence/entropy : 이중 어떤 signal이 더 구분하기 좋은가
    TPR@1%FPR : 오탐률 1% 이하에서 member을 얼마나 잘 잡는지
    precision/recall (threshold 기반 보조 지표)
    """

    auc_loss   = _run_auc(loss_scores_m, loss_scores_nm)
    auc_conf   = _run_auc(member_signals["confidence"], nonmember_signals["confidence"])
    auc_entr   = _run_auc(-member_signals["entropy"], -nonmember_signals["entropy"])
    auc_margin   = _run_auc(member_signals["margin"], nonmember_signals["margin"])
    auc_aug_mean = _run_auc(-member_signals["aug_loss_mean"], -nonmember_signals["aug_loss_mean"])
    auc_aug_std  = _run_auc(-member_signals["aug_loss_std"],  -nonmember_signals["aug_loss_std"])

    loss_scores = np.concatenate([loss_scores_m, loss_scores_nm])
    fpr, tpr, _ = roc_curve(labels, loss_scores)

    valid_idx = np.where(fpr <= 0.01)[0]
    tpr_at_1fpr = float(tpr[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0

    # [수정 위치 9] threshold는 attack validation에서 선택하고, 여기서는 attack test에만 적용
    preds = (loss_scores >= threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    loss_gap        = float(np.mean(nonmember_signals["losses"]) - np.mean(member_signals["losses"]))
    conf_gap        = float(np.mean(member_signals["confidence"]) - np.mean(nonmember_signals["confidence"]))
    entr_gap        = float(np.mean(nonmember_signals["entropy"]) - np.mean(member_signals["entropy"]))
    margin_gap      = float(np.mean(member_signals["margin"]) - np.mean(nonmember_signals["margin"]))
    _, loss_separation_d = standardized_gap(loss_scores_m, loss_scores_nm)
    attack_advantage = float(np.max(tpr - fpr))

    return {
        "auc_loss": auc_loss,
        "auc_conf": auc_conf,
        "auc_entr": auc_entr,
        "auc_margin": auc_margin,
        "auc_aug_mean": auc_aug_mean,
        "auc_aug_std":  auc_aug_std,
        "member_aug_std":    member_signals["aug_loss_std"],
        "nonmember_aug_std": nonmember_signals["aug_loss_std"],
        "tpr_at_1fpr": tpr_at_1fpr,
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "fpr": fpr,
        "tpr": tpr,
        "member_losses": member_signals["losses"],
        "nonmember_losses": nonmember_signals["losses"],
        "member_confidences":    member_signals["confidence"],
        "nonmember_confidences": nonmember_signals["confidence"],
        "member_entropies":      member_signals["entropy"],
        "nonmember_entropies":   nonmember_signals["entropy"],
        "member_margins":        member_signals["margin"],
        "nonmember_margins":     nonmember_signals["margin"],
        "loss_gap": loss_gap,
        "conf_gap": conf_gap,
        "entr_gap": entr_gap,
        "margin_gap": margin_gap,
        "loss_separation_d": loss_separation_d,
        "attack_advantage": attack_advantage,
    }

# ── XGB Attack ───────────────────────────────────────────────────────────────
def stack_xgb_features(member_signals, nonmember_signals):
    X_member = np.column_stack([
        -member_signals["losses"],
        -member_signals["aug_loss_mean"],
        member_signals["aug_loss_mean"] - member_signals["losses"],
        member_signals["aug_loss_std"],
    ])
    X_nonmember = np.column_stack([
        -nonmember_signals["losses"],
        -nonmember_signals["aug_loss_mean"],
        nonmember_signals["aug_loss_mean"] - nonmember_signals["losses"],
        nonmember_signals["aug_loss_std"],
    ])
    X = np.vstack([X_member, X_nonmember])
    y = np.concatenate([np.ones(len(X_member)), np.zeros(len(X_nonmember))])
    return X, y


def run_xgb_attack(member_val_signals, nonmember_val_signals, member_test_signals, nonmember_test_signals, seed=SEED):
    """
    XGBoost learned attack.

    Train split : attack-validation member/non-member signals
    Test split  : attack-test member/non-member signals
    Threshold   : Youden J on val ROC (not test) — leakage 방지
    Reported   : AUC_XGB, TPR@1%FPR_XGB, Adv_XGB, precision_xgb, recall_xgb, separation d
    """
    X_val, y_val = stack_xgb_features(member_val_signals, nonmember_val_signals)
    X_test, y_test = stack_xgb_features(member_test_signals, nonmember_test_signals)

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(X_val, y_val)

    val_scores = clf.predict_proba(X_val)[:, 1]
    val_fpr, val_tpr, val_thresholds = roc_curve(y_val, val_scores)
    best_threshold = float(val_thresholds[int(np.argmax(val_tpr - val_fpr))])

    scores = clf.predict_proba(X_test)[:, 1]
    auc = _safe_auc(y_test, scores)
    fpr, tpr, _ = roc_curve(y_test, scores)
    valid_idx = np.where(fpr <= 0.01)[0]
    tpr_at_1fpr = float(tpr[valid_idx[-1]]) if len(valid_idx) > 0 else 0.0
    advantage = float(np.max(tpr - fpr))

    preds = (scores >= best_threshold).astype(int)
    precision_xgb = float(precision_score(y_test, preds, zero_division=0))
    recall_xgb    = float(recall_score(y_test, preds, zero_division=0))

    member_xgb_scores    = scores[y_test == 1]
    nonmember_xgb_scores = scores[y_test == 0]
    xgb_score_gap, xgb_separation_d = standardized_gap(member_xgb_scores, nonmember_xgb_scores)

    return {
        "auc_xgb": auc,
        "tpr_at_1fpr_xgb": tpr_at_1fpr,
        "attack_advantage_xgb": advantage,
        "threshold_xgb": best_threshold,
        "precision_xgb": precision_xgb,
        "recall_xgb": recall_xgb,
        "fpr_xgb": fpr,
        "tpr_xgb": tpr,
        "xgb_scores": scores,
        "xgb_labels": y_test,
        "member_xgb_scores": member_xgb_scores,
        "nonmember_xgb_scores": nonmember_xgb_scores,
        "xgb_score_gap": xgb_score_gap,
        "xgb_separation_d": xgb_separation_d,
    }


# ── Run One Experiment ────────────────────────────────────────────────────────

def _prepare_splits(data_seed, n, member_order, nonmember_order, fake_order):
    sampled_members    = member_order[:n]
    sampled_nonmembers = nonmember_order[:n]
    sampled_fakes      = fake_order[:n]

    base_seed   = SEED + data_seed * 10000 + n * 10
    attack_seed = base_seed + 3

    real_train_files, real_val_files = split_files_train_val_nested(sampled_members, TRAIN_VAL_RATIO)
    fake_train_files, fake_val_files = split_files_train_val_nested(sampled_fakes, TRAIN_VAL_RATIO)
    attack_files = split_attack_files(real_train_files, sampled_nonmembers, ATTACK_VAL_RATIO, attack_seed)

    print("    Train/val and attack splits are fixed across all model runs.")
    print(f"    Real train members: {len(real_train_files)}, real val excluded from MIA: {len(real_val_files)}")
    print(f"    Fake train: {len(fake_train_files)}, fake val: {len(fake_val_files)}")
    print(
        f"    Attack val: M={len(attack_files['member_attack_val'])}, NM={len(attack_files['nonmember_attack_val'])} | "
        f"Attack test: M={len(attack_files['member_attack_test'])}, NM={len(attack_files['nonmember_attack_test'])}"
    )
    return {
        "real_train_files": real_train_files,
        "real_val_files":   real_val_files,
        "fake_train_files": fake_train_files,
        "fake_val_files":   fake_val_files,
        "attack_files":     attack_files,
        "base_seed":        base_seed,
        "attack_seed":      attack_seed,
    }


def _run_single_model(splits, run_idx, data_seed, n):
    real_train_files = splits["real_train_files"]
    real_val_files   = splits["real_val_files"]
    fake_train_files = splits["fake_train_files"]
    fake_val_files   = splits["fake_val_files"]
    attack_files     = splits["attack_files"]
    base_seed        = splits["base_seed"]
    attack_seed      = splits["attack_seed"]

    model_seed  = base_seed + run_idx
    loader_seed = model_seed + 2

    print(f"\n  [Model Run {run_idx + 1}/{NUM_MODEL_RUNS}]  model_seed={model_seed}")
    set_global_seed(model_seed)

    member_real_train_ds     = FaceDataset(real_train_files, label=0, transform=transform)
    member_real_val_ds       = FaceDataset(real_val_files,   label=0, transform=transform)
    fake_train_ds            = FaceDataset(fake_train_files, label=1, transform=transform)
    fake_val_ds              = FaceDataset(fake_val_files,   label=1, transform=transform)
    attack_member_val_ds     = FaceDataset(attack_files["member_attack_val"],     label=0, transform=transform)
    attack_member_test_ds    = FaceDataset(attack_files["member_attack_test"],    label=0, transform=transform)
    attack_nonmember_val_ds  = FaceDataset(attack_files["nonmember_attack_val"],  label=0, transform=transform)
    attack_nonmember_test_ds = FaceDataset(attack_files["nonmember_attack_test"], label=0, transform=transform)

    train_gen    = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(
        ConcatDataset([member_real_train_ds, fake_train_ds]),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=train_gen,
    )
    val_loader = DataLoader(
        ConcatDataset([member_real_val_ds, fake_val_ds]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CHECKPOINT_DIR / (
        f"ffhq_mia_resnet34_ds{data_seed}_n{n}_run{run_idx + 1}_"
        f"ep{EPOCHS}_bs{BATCH_SIZE}_seed{model_seed}.pth"
    )

    if USE_SAVED_MODELS and model_path.exists():
        model = build_model()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        best_train_loss = best_val_loss = gen_gap = None
        print(f"      Loaded saved model: {model_path}")
    else:
        model = build_model()
        print("      Training ResNet34...")
        best_val_loss, best_train_loss = train_model(model, train_loader, EPOCHS, val_loader)
        gen_gap = best_val_loss - best_train_loss
        torch.save(model.state_dict(), model_path)
        print(
            f"      Saved model: {model_path}  "
            f"best_train_loss={best_train_loss:.4f}  best_val_loss={best_val_loss:.4f}  gen_gap={gen_gap:.4f}"
        )

    print("      Computing attack-validation signals...")
    member_val_signals    = compute_signals(model, attack_member_val_ds,    aug_seed=attack_seed)
    nonmember_val_signals = compute_signals(model, attack_nonmember_val_ds, aug_seed=attack_seed)
    threshold = choose_threshold_on_attack_val(member_val_signals, nonmember_val_signals)

    print("      Computing attack-test signals...")
    member_test_signals    = compute_signals(model, attack_member_test_ds,    aug_seed=attack_seed)
    nonmember_test_signals = compute_signals(model, attack_nonmember_test_ds, aug_seed=attack_seed)

    res = evaluate_mia(member_test_signals, nonmember_test_signals, threshold)
    res.update(run_xgb_attack(
        member_val_signals, nonmember_val_signals,
        member_test_signals, nonmember_test_signals,
        seed=attack_seed,
    ))
    res.update({
        "data_seed": data_seed, "subset_size": n, "model_run": run_idx + 1,
        "model_seed": model_seed,
        "n_train_member":       len(member_real_train_ds),
        "n_val_real_excluded":  len(member_real_val_ds),
        "n_attack_val_member":  len(attack_member_val_ds),
        "n_attack_test_member": len(attack_member_test_ds),
        "best_train_loss": best_train_loss, "best_val_loss": best_val_loss, "gen_gap": gen_gap,
    })
    print(
        f"      AUC_loss={res['auc_loss']:.4f}  AUC_XGB={res['auc_xgb']:.4f}  "
        f"ΔAUC={res['auc_xgb'] - res['auc_loss']:+.4f}  "
        f"TPR@1%FPR_loss={res['tpr_at_1fpr']:.4f}  TPR@1%FPR_XGB={res['tpr_at_1fpr_xgb']:.4f}  "
        f"ΔTPR={res['tpr_at_1fpr_xgb'] - res['tpr_at_1fpr']:+.4f}  "
        f"Adv_loss={res['attack_advantage']:.4f}  Adv_XGB={res['attack_advantage_xgb']:.4f}  "
        f"ΔAdv={res['attack_advantage_xgb'] - res['attack_advantage']:+.4f}  "
        f"AUC_conf={res['auc_conf']:.4f}  AUC_entr={res['auc_entr']:.4f}  AUC_margin={res['auc_margin']:.4f}  "
        f"Precision={res['precision']:.4f}  Recall={res['recall']:.4f}"
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def _aggregate_runs(run_results, data_seed, n):
    aucs               = _collect(run_results, "auc_loss")
    tprs               = _collect(run_results, "tpr_at_1fpr")
    aucs_xgb           = _collect(run_results, "auc_xgb")
    tprs_xgb           = _collect(run_results, "tpr_at_1fpr_xgb")
    advs_xgb           = _collect(run_results, "attack_advantage_xgb")
    attack_advantages  = _collect(run_results, "attack_advantage")
    aucs_margin        = _collect(run_results, "auc_margin")
    margin_gaps        = _collect(run_results, "margin_gap")
    loss_separation_ds = _collect(run_results, "loss_separation_d")

    ci = stats.t.interval(0.95, df=len(aucs) - 1, loc=np.mean(aucs), scale=stats.sem(aucs)) if len(aucs) >= 2 else (np.nan, np.nan)

    print(f"\n  --- Per-model-run summary (data_seed={data_seed}, n={n}) ---")
    for i, r in enumerate(run_results):
        print(
            f"  Run {i + 1}: "
            f"AUC_loss={r['auc_loss']:.4f}  AUC_XGB={r['auc_xgb']:.4f}  "
            f"ΔAUC={r['auc_xgb'] - r['auc_loss']:+.4f}  "
            f"TPR_loss={r['tpr_at_1fpr']:.4f}  TPR_XGB={r['tpr_at_1fpr_xgb']:.4f}  "
            f"ΔTPR={r['tpr_at_1fpr_xgb'] - r['tpr_at_1fpr']:+.4f}  "
            f"Adv_loss={r['attack_advantage']:.4f}  Adv_XGB={r['attack_advantage_xgb']:.4f}  "
            f"margin_gap={r['margin_gap']:.4f}  loss_sep_d={r['loss_separation_d']:.4f}  "
            f"Prec_XGB={r['precision_xgb']:.4f}  Rec_XGB={r['recall_xgb']:.4f}"
        )
    print(
        f"  Average: AUC_loss={np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
        f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
        f"AUC_XGB={np.mean(aucs_xgb):.4f}  ΔAUC={np.mean(aucs_xgb) - np.mean(aucs):+.4f}  "
        f"TPR_loss={np.mean(tprs):.4f}  TPR_XGB={np.mean(tprs_xgb):.4f}  "
        f"ΔTPR={np.mean(tprs_xgb) - np.mean(tprs):+.4f}  "
        f"Adv_loss={np.mean(attack_advantages):.4f}  Adv_XGB={np.mean(advs_xgb):.4f}  "
        f"ΔAdv={np.mean(advs_xgb) - np.mean(attack_advantages):+.4f}  "
        f"AUC_margin={np.mean(aucs_margin):.4f}  margin_gap={np.mean(margin_gaps):.4f}  "
        f"loss_sep_d={np.mean(loss_separation_ds):.4f}"
    )

    median_run_idx = int(np.argsort(aucs)[len(aucs) // 2])
    median_res     = run_results[median_run_idx]

    return {
        "data_seed":    data_seed,
        "subset_size":  n,
        "auc_loss":     float(np.mean(aucs)),
        "auc_loss_std": float(np.std(aucs)),
        "auc_conf":     _mean(run_results, "auc_conf"),
        "auc_entr":     _mean(run_results, "auc_entr"),
        "auc_margin":   float(np.mean(aucs_margin)),
        "auc_xgb":      float(np.mean(aucs_xgb)),
        "tpr_at_1fpr_xgb":      float(np.mean(tprs_xgb)),
        "attack_advantage_xgb": float(np.mean(advs_xgb)),
        "auc_aug_mean":  _mean(run_results, "auc_aug_mean"),
        "auc_aug_std":   _mean(run_results, "auc_aug_std"),
        "member_aug_std":    median_res["member_aug_std"],
        "nonmember_aug_std": median_res["nonmember_aug_std"],
        "ci_low":  float(ci[0]),
        "ci_high": float(ci[1]),
        "tpr_at_1fpr": float(np.mean(tprs)),
        "tpr_std":     float(np.std(tprs)),
        "precision":     _mean(run_results, "precision"),
        "recall":        _mean(run_results, "recall"),
        "precision_xgb": _mean(run_results, "precision_xgb"),
        "recall_xgb":    _mean(run_results, "recall_xgb"),
        "fpr":     median_res["fpr"],
        "tpr":     median_res["tpr"],
        "fpr_xgb": median_res["fpr_xgb"],
        "tpr_xgb": median_res["tpr_xgb"],
        "xgb_scores": median_res["xgb_scores"],
        "xgb_labels": median_res["xgb_labels"],
        "member_xgb_scores":    median_res["member_xgb_scores"],
        "nonmember_xgb_scores": median_res["nonmember_xgb_scores"],
        "member_losses":         median_res["member_losses"],
        "nonmember_losses":      median_res["nonmember_losses"],
        "member_confidences":    median_res["member_confidences"],
        "nonmember_confidences": median_res["nonmember_confidences"],
        "member_entropies":      median_res["member_entropies"],
        "nonmember_entropies":   median_res["nonmember_entropies"],
        "member_margins":        median_res["member_margins"],
        "nonmember_margins":     median_res["nonmember_margins"],
        "runs":              run_results,
        "loss_gap":          _mean(run_results, "loss_gap"),
        "conf_gap":          _mean(run_results, "conf_gap"),
        "entr_gap":          _mean(run_results, "entr_gap"),
        "margin_gap":        float(np.mean(margin_gaps)),
        "loss_separation_d": float(np.mean(loss_separation_ds)),
        "xgb_score_gap":     _mean(run_results, "xgb_score_gap"),
        "xgb_separation_d":  _mean(run_results, "xgb_separation_d"),
        "attack_advantage":  float(np.mean(attack_advantages)),
        "gen_gap":           _mean(run_results, "gen_gap"),
    }


def run_experiment(data_seed, subset_size, member_order, nonmember_order, fake_order):
    n = subset_size
    if len(member_order) < n or len(fake_order) < n or len(nonmember_order) < n:
        raise ValueError(f"Not enough files for subset_size={n}.")

    print(f"\n{'=' * 72}")
    print(f"  Data seed: {data_seed}  |  Subset n={n}  |  Epochs: {EPOCHS}  |  Model runs: {NUM_MODEL_RUNS}")
    print(f"{'=' * 72}")

    splits = _prepare_splits(data_seed, n, member_order, nonmember_order, fake_order)
    run_results = [_run_single_model(splits, run_idx, data_seed, n) for run_idx in range(NUM_MODEL_RUNS)]
    return _aggregate_runs(run_results, data_seed, n)

# ── Aggregation ───────────────────────────────────────────────────────────────
# Results aggregation: model run 평균과 data seed 평균을 분리
def summarize_across_data_seeds(all_results):
    summary = {}
    for n in SUBSET_SIZES:
        seed_results = [res for (data_seed, subset_size), res in all_results.items() if subset_size == n]
        if not seed_results:
            continue

        aucs     = _collect(seed_results, "auc_loss")
        aucs_xgb = _collect(seed_results, "auc_xgb")

        ci     = stats.t.interval(0.95, df=len(aucs) - 1,     loc=np.mean(aucs),     scale=stats.sem(aucs))     if len(aucs)     >= 2 else (np.nan, np.nan)
        ci_xgb = stats.t.interval(0.95, df=len(aucs_xgb) - 1, loc=np.mean(aucs_xgb), scale=stats.sem(aucs_xgb)) if len(aucs_xgb) >= 2 else (np.nan, np.nan)

        median_idx = int(np.argsort(aucs)[len(aucs) // 2])
        median_res = seed_results[median_idx]

        summary[n] = {
            "subset_size":                    n,
            "auc_loss":                       float(np.mean(aucs)),
            "auc_loss_std_across_data_seeds": float(np.std(aucs)),
            "auc_conf":                       _mean(seed_results, "auc_conf"),
            "auc_entr":                       _mean(seed_results, "auc_entr"),
            "ci_low_across_data_seeds":       float(ci[0]),
            "ci_high_across_data_seeds":      float(ci[1]),
            "tpr_at_1fpr":                    _mean(seed_results, "tpr_at_1fpr"),
            "precision":                      _mean(seed_results, "precision"),
            "recall":                         _mean(seed_results, "recall"),
            "precision_xgb":                  _mean(seed_results, "precision_xgb"),
            "recall_xgb":                     _mean(seed_results, "recall_xgb"),
            "gen_gap":                        _mean(seed_results, "gen_gap"),
            "fpr":                            median_res["fpr"],
            "tpr":                            median_res["tpr"],
            "member_losses":                  median_res["member_losses"],
            "nonmember_losses":               median_res["nonmember_losses"],
            "member_confidences":    median_res["member_confidences"],
            "nonmember_confidences": median_res["nonmember_confidences"],
            "member_entropies":      median_res["member_entropies"],
            "nonmember_entropies":   median_res["nonmember_entropies"],
            "member_margins":        median_res["member_margins"],
            "nonmember_margins":     median_res["nonmember_margins"],
            "member_xgb_scores":     median_res["member_xgb_scores"],
            "nonmember_xgb_scores":  median_res["nonmember_xgb_scores"],
            "loss_gap":    _mean(seed_results, "loss_gap"),
            "conf_gap":    _mean(seed_results, "conf_gap"),
            "entr_gap":    _mean(seed_results, "entr_gap"),
            "auc_margin":  _mean(seed_results, "auc_margin"),
            "auc_xgb":                       float(np.mean(aucs_xgb)),
            "auc_xgb_std_across_data_seeds": float(np.std(aucs_xgb)),
            "xgb_ci_low_across_data_seeds":  float(ci_xgb[0]),
            "xgb_ci_high_across_data_seeds": float(ci_xgb[1]),
            "tpr_at_1fpr_xgb":      _mean(seed_results, "tpr_at_1fpr_xgb"),
            "attack_advantage_xgb": _mean(seed_results, "attack_advantage_xgb"),
            "delta_auc_xgb_loss":   float(np.mean(aucs_xgb) - np.mean(aucs)),
            "delta_tpr_xgb_loss":   float(_mean(seed_results, "tpr_at_1fpr_xgb") - _mean(seed_results, "tpr_at_1fpr")),
            "delta_adv_xgb_loss":   float(_mean(seed_results, "attack_advantage_xgb") - _mean(seed_results, "attack_advantage")),
            "auc_aug_mean": _mean(seed_results, "auc_aug_mean"),
            "auc_aug_std":  _mean(seed_results, "auc_aug_std"),
            "member_aug_std":    median_res["member_aug_std"],
            "nonmember_aug_std": median_res["nonmember_aug_std"],
            "fpr_xgb": median_res["fpr_xgb"],
            "tpr_xgb": median_res["tpr_xgb"],
            "margin_gap":        _mean(seed_results, "margin_gap"),
            "loss_separation_d": _mean(seed_results, "loss_separation_d"),
            "xgb_score_gap":     _mean(seed_results, "xgb_score_gap"),
            "xgb_separation_d":  _mean(seed_results, "xgb_separation_d"),
            "attack_advantage":  _mean(seed_results, "attack_advantage"),
        }
    return summary

# ── Visualization ─────────────────────────────────────────────────────────────
# Plot_loss_distribution: member/non-member real loss 분포
# => 두 분포가 잘 분리되면, loss 기반 MIA 가능성 있음
def plot_loss_distributions(summary):
    n_plots = len(summary)
    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 6), sharey=False)
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    signal_keys = [
        ("member_losses",      "nonmember_losses",      "Cross-Entropy Loss"),
        ("member_confidences", "nonmember_confidences", "True-class Confidence"),
    ]

    for col, subset_size in enumerate(sorted(summary.keys())):
        res = summary[subset_size]
        for row, (mk, nmk, xlabel) in enumerate(signal_keys):
            ax = axes[row][col]
            ax.hist(res[mk],  bins=25, alpha=0.65, label="Member (train)",       color="steelblue", density=True)
            ax.hist(res[nmk], bins=25, alpha=0.65, label="Non-member (holdout)", color="tomato",    density=True)
            if row == 0:
                ax.set_title(f"n={subset_size}", fontsize=11)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.legend(fontsize=7)

    fig.suptitle("Signal Distribution: Member vs Non-member (FFHQ real only)", y=1.01, fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_roc_curves(summary):
    """Loss ROC와 XGB ROC를 같은 subset size별로 비교."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = sorted(summary.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sizes)))

    for subset_size, color in zip(sizes, colors):
        res = summary[subset_size]
        ax.plot(
            res["fpr"], res["tpr"],
            color=color, lw=2, linestyle="-",
            label=f"Loss n={subset_size} AUC={res['auc_loss']:.3f}"
        )
        ax.plot(
            res["fpr_xgb"], res["tpr_xgb"],
            color=color, lw=2, linestyle="--",
            label=f"XGB n={subset_size} AUC={res['auc_xgb']:.3f}"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random AUC=0.500")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Loss ROC vs XGB ROC", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_auc_by_size(summary):
    """Loss AUC vs XGB AUC, Loss TPR@1%FPR vs XGB TPR@1%FPR 비교."""
    sizes = sorted(summary.keys())
    xlabels = [f"n={n}" for n in sizes]

    auc_loss = [summary[n]["auc_loss"]       for n in sizes]
    auc_xgb  = [summary[n]["auc_xgb"]        for n in sizes]
    tpr_loss = [summary[n]["tpr_at_1fpr"]     for n in sizes]
    tpr_xgb  = [summary[n]["tpr_at_1fpr_xgb"] for n in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(sizes, auc_loss, marker="o", lw=2, ms=8, label="Loss AUC")
    axes[0].plot(sizes, auc_xgb,  marker="s", lw=2, ms=8, label="XGB AUC")
    axes[0].axhline(0.5, color="gray", linestyle="--", lw=1, label="Random baseline")
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels(xlabels, fontsize=9)
    axes[0].set_xlabel("Subset Size", fontsize=11)
    axes[0].set_ylabel("AUC", fontsize=11)
    axes[0].set_title("Loss AUC vs XGB AUC", fontsize=12)
    axes[0].set_ylim(0.4, 1.05)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(sizes, tpr_loss, marker="o", lw=2, ms=8, label="Loss TPR@1%FPR")
    axes[1].plot(sizes, tpr_xgb,  marker="s", lw=2, ms=8, label="XGB TPR@1%FPR")
    axes[1].axhline(0.01, color="gray", linestyle="--", lw=1, label="Random baseline")
    axes[1].set_xticks(sizes)
    axes[1].set_xticklabels(xlabels, fontsize=9)
    axes[1].set_xlabel("Subset Size", fontsize=11)
    axes[1].set_ylabel("TPR @ 1% FPR", fontsize=11)
    axes[1].set_title("Loss TPR@1%FPR vs XGB TPR@1%FPR", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    top = max(tpr_loss + tpr_xgb) if (tpr_loss + tpr_xgb) else 0.1
    axes[1].set_ylim(0, top * 1.3 + 0.02)

    fig.suptitle("Loss Attack vs XGBoost Attack", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.show()

def plot_signal_gaps(summary):
    sizes     = list(summary.keys())
    loss_gaps = [summary[n]["loss_gap"] for n in sizes]
    conf_gaps = [summary[n]["conf_gap"] for n in sizes]
    entr_gaps = [summary[n]["entr_gap"] for n in sizes]
    xlabels   = [f"n={n}" for n in sizes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, loss_gaps, marker="o", lw=2, label="Loss gap")
    ax.plot(sizes, conf_gaps, marker="s", lw=2, label="Confidence gap")
    ax.plot(sizes, entr_gaps, marker="^", lw=2, label="Entropy gap")
    ax.axhline(0, color="gray", linestyle="--", lw=1)
    ax.set_xticks(sizes)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Subset Size", fontsize=11)
    ax.set_ylabel("Member / Non-member Signal Gap", fontsize=11)
    ax.set_title("Signal Gaps by Training Set Size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_gap_vs_auc(summary):
    sizes     = list(summary.keys())
    loss_gaps = [summary[n]["loss_gap"] for n in sizes]
    conf_gaps = [summary[n]["conf_gap"] for n in sizes]
    entr_gaps = [summary[n]["entr_gap"] for n in sizes]
    auc_loss  = [summary[n]["auc_loss"] for n in sizes]
    auc_conf  = [summary[n]["auc_conf"] for n in sizes]
    auc_entr  = [summary[n]["auc_entr"] for n in sizes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_specs = [
        (axes[0], loss_gaps, auc_loss, "Loss Gap",       "AUC_loss"),
        (axes[1], conf_gaps, auc_conf, "Confidence Gap", "AUC_conf"),
        (axes[2], entr_gaps, auc_entr, "Entropy Gap",    "AUC_entr"),
    ]
    for ax, gaps, aucs, xlabel, ylabel in plot_specs:
        ax.scatter(gaps, aucs, s=80, color="steelblue")
        for i, n in enumerate(sizes):
            ax.annotate(f"n={n}", (gaps[i], aucs[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random baseline")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{xlabel} vs {ylabel}", fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Signal Gap vs MIA AUC", fontsize=13, y=1.04)
    plt.tight_layout()
    plt.show()

def plot_generalization_gap(summary):
    valid_items = [(n, res) for n, res in summary.items() if res.get("gen_gap") is not None]
    if not valid_items:
        print("No generalization gap values available. Skipping gap plot.")
        return
    sizes   = [n for n, _ in valid_items]
    gaps    = [res["gen_gap"] for _, res in valid_items]
    aucs    = [res["auc_loss"] for _, res in valid_items]
    xlabels = [f"n={n}" for n in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(sizes, gaps, marker="o", color="purple", lw=2, ms=8)
    axes[0].axhline(0, color="gray", linestyle="--", lw=1)
    axes[0].set_xticks(sizes)
    axes[0].set_xticklabels(xlabels, fontsize=9)
    axes[0].set_xlabel("Subset Size", fontsize=11)
    axes[0].set_ylabel("Generalization Gap (val - train loss)", fontsize=11)
    axes[0].set_title("Generalization Gap vs Subset Size", fontsize=12)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(gaps, aucs, color="steelblue", s=80)
    for i, n in enumerate(sizes):
        axes[1].annotate(f"n={n}", (gaps[i], aucs[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].axhline(0.5, color="gray", linestyle="--", lw=1, label="Random baseline")
    axes[1].set_xlabel("Generalization Gap", fontsize=11)
    axes[1].set_ylabel("AUC (Loss Attack)", fontsize=11)
    axes[1].set_title("Generalization Gap vs MIA AUC", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Generalization Gap Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

# ── Results Table ─────────────────────────────────────────────────────────────
def print_results_table(summary):
    print("\n" + "=" * 240)
    print(
        f"  {'n_real':>8} | {'AUC_loss':>8} | {'AUC_XGB':>8} | {'ΔAUC':>8} | "
        f"{'+-data':>8} | {'95% CI low':>10} | {'95% CI high':>11} | "
        f"{'TPR_loss':>10} | {'TPR_XGB':>10} | {'ΔTPR':>8} | "
        f"{'Adv_loss':>8} | {'Adv_XGB':>8} | {'ΔAdv':>8} | "
        f"{'AUC_margin':>10} | {'AUC_augM':>9} | {'AUC_augS':>9} | "
        f"{'Precision':>10} | {'Recall':>8} | "
        f"{'Prec_XGB':>9} | {'Rec_XGB':>8} | "
        f"{'loss_sep_d':>10} | {'xgb_sep_d':>9}"
    )
    print("─" * 240)
    for subset_size in sorted(summary.keys()):
        res = summary[subset_size]
        print(
            f"  {subset_size:>8} | "
            f"{res['auc_loss']:>8.4f} | "
            f"{res['auc_xgb']:>8.4f} | "
            f"{res['delta_auc_xgb_loss']:>+8.4f} | "
            f"{res['auc_loss_std_across_data_seeds']:>8.4f} | "
            f"{res['ci_low_across_data_seeds']:>10.4f} | "
            f"{res['ci_high_across_data_seeds']:>11.4f} | "
            f"{res['tpr_at_1fpr']:>10.4f} | "
            f"{res['tpr_at_1fpr_xgb']:>10.4f} | "
            f"{res['delta_tpr_xgb_loss']:>+8.4f} | "
            f"{res['attack_advantage']:>8.4f} | "
            f"{res['attack_advantage_xgb']:>8.4f} | "
            f"{res['delta_adv_xgb_loss']:>+8.4f} | "
            f"{res['auc_margin']:>10.4f} | "
            f"{res['auc_aug_mean']:>9.4f} | "
            f"{res['auc_aug_std']:>9.4f} | "
            f"{res['precision']:>10.4f} | "
            f"{res['recall']:>8.4f} | "
            f"{res['precision_xgb']:>9.4f} | "
            f"{res['recall_xgb']:>8.4f} | "
            f"{res['loss_separation_d']:>10.4f} | "
            f"{res['xgb_separation_d']:>9.4f}"
        )
    print("=" * 240)

def print_gap_table(summary):
    print("\n" + "=" * 110)
    print(
        f"  {'n_real':>8} | {'loss_gap':>10} | {'conf_gap':>10} | {'entr_gap':>10} | "
        f"{'gen_gap':>10} | {'AUC_loss':>8} | {'AUC_conf':>8} | {'AUC_entr':>8}"
    )
    print("-" * 110)
    for subset_size, res in summary.items():
        gen_gap = res.get("gen_gap")
        gen_gap_str = f"{gen_gap:.4f}" if gen_gap is not None else "NA"
        print(
            f"  {subset_size:>8} | "
            f"{res['loss_gap']:>10.4f} | "
            f"{res['conf_gap']:>10.4f} | "
            f"{res['entr_gap']:>10.4f} | "
            f"{gen_gap_str:>10} | "
            f"{res['auc_loss']:>8.4f} | "
            f"{res['auc_conf']:>8.4f} | "
            f"{res['auc_entr']:>8.4f}"
        )
    print("=" * 110)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU    : {props.name}")
        print(f"VRAM   : {props.total_memory / 1024**3:.1f} GB")

    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    real_files = sorted(
        os.path.join(REAL_PATH, f) for f in os.listdir(REAL_PATH)
        if os.path.splitext(f)[1] in exts
    )
    fake_files = sorted(
        os.path.join(FAKE_PATH, f) for f in os.listdir(FAKE_PATH)
        if os.path.splitext(f)[1] in exts
    )
    print(f"Real (FFHQ): {len(real_files)}, Fake (StyleGAN): {len(fake_files)} files")

    if not real_files or not fake_files:
        raise RuntimeError("REAL_PATH/FAKE_PATH에서 이미지 파일을 찾지 못했습니다.")

    all_results = {}
    for data_seed in DATA_SEEDS:
        print(f"\n\n######## DATA SEED {data_seed} ########")
        member_order, nonmember_order, fake_order = make_nested_orders(real_files, fake_files, data_seed)
        print(f"member_pool: {len(member_order)}, nonmember_pool: {len(nonmember_order)}, fake_pool: {len(fake_order)}")

        for subset_size in SUBSET_SIZES:
            all_results[(data_seed, subset_size)] = run_experiment(
                data_seed=data_seed,
                subset_size=subset_size,
                member_order=member_order,
                nonmember_order=nonmember_order,
                fake_order=fake_order,
            )

    summary = summarize_across_data_seeds(all_results)
    print_results_table(summary)

    print("\nGenerating plots...")
    plot_loss_distributions(summary)
    plot_roc_curves(summary)      # Loss ROC + XGB ROC
    plot_auc_by_size(summary)     # Loss vs XGB AUC/TPR comparison
    plot_generalization_gap(summary)
    print_gap_table(summary)
    plot_signal_gaps(summary)
    plot_gap_vs_auc(summary)

    print("\nExperiment complete.")

    return all_results, summary

if __name__ == "__main__":
    all_results, summary = main()