# 分支与版本管理规范

> 本仓库是 LHAF / MMER-Adapter 多模态情感分析实验代码库。
> **`main` 是唯一常青分支，历史版本一律由 tag 冻结。** 本规范自 2026-10-18 仓库整理后生效。

## 1. 为什么不再用「版本号分支」

整理前仓库有 13 条远程分支、0 个 tag，`main` 自 2026-04 起停滞。实测各分支相对 `LHAv1.0` 的独有提交数：

```
HMMEM / HMMEM2.0 / HAMA-DBMv1.0 / ver0.1 / ver0.2   0 个
HMMEM1.1 3 个、MoE0.3 3 个、MoE0.3.1 2 个、MoE0.4 1 个（均为已被重写的旧实现或日志文件）
main     5 个（一次 PR 合并 + 三重 Revert 链）
```

也就是说：**12 条分支没有一条携带未吸收的独立代码**，它们只是同一棵树在不同时间点的快照。用长期分支承载快照，会导致「哪个分支才是最新的」这种问题反复出现，而 tag 天生只读、不可误提交、`git tag -l` 一目了然，是快照的正确载体。

结论：分支只保留 1~3 条短命工作线，版本交给 tag。

## 2. 分支模型

| 分支 | 用途 | 生命周期 | 准入条件 |
|---|---|---|---|
| `main` | 可运行主线 | 永久 | `python -m pytest tests/ -q` 全绿 + `python run.py --dry_run --print_config` 正常输出 |
| `exp/<模型>-<数据集>-<yyyymm>` | 单次实验 | ≤ 2 周，出结论即删 | 自由提交，不要求全绿 |
| `next/<主题>` | 大版本重构（如 `next/v3`） | 合并进 `main` 后删除 | 合并前需在 `main` 上跑通测试与一次最小实验 |
| `hotfix/<问题>` | 线上/论文复现紧急修复 | 合并后删除 | 同 `main` |

硬性约束：

- **禁止用版本号命名分支**。`v1.0`、`v2.0` 是 tag 的名字，不是分支的名字。
- **禁止长期保留「我的实验线」分支**。实验线做完就打 tag 删分支，否则半年后没人分得清哪条是活的。
- **禁止直接 push 到 `main`**（单人也要开 PR 或至少本地跑完准入检查），`main` 上的每个提交都应可运行。

## 3. 日常工作流

```powershell
# 1) 从最新主线开实验分支
git switch main
git pull --ff-only
git switch -c exp/hmmem-meld-202610

# 2) 实验过程中的提交（可以有 WIP）
git add -A
git commit -m "train: 新增 qwen3.5-4B 在 meld 上的全量实验"

# 3) 出结论后，要么合并回 main，要么打 tag 留档后删除
git push -u origin exp/hmmem-meld-202610

# 4a) 值得进主线：走 PR，准入检查通过后 squash 合并
# 4b) 只是留档：
git tag -a exp/hmmem-meld-202610-f1 -m "MELD F1=0.xxxx，用于论文表 3"
git push origin exp/hmmem-meld-202610-f1
git push origin --delete exp/hmmem-meld-202610
git branch -d exp/hmmem-meld-202610
```

## 4. tag 规范

| 前缀 | 含义 | 规则 |
|---|---|---|
| `v<major>.<minor>` | 里程碑版本（论文/对外发布对应点） | 必须 annotated（`-a` + 说明），**禁止移动或删除** |
| `archive/<名称>` | 历史分支快照，只读 | 不得再创建同名分支 |
| `exp/<名称>` | 单次实验可复现快照 | 用于论文数字溯源，说明里写清指标 |

```powershell
git tag -a v2.1 -m "LHAF v2.1：新增 xxx"
git push origin v2.1
git tag -l -n1          # 查看带说明的 tag 列表
```

## 5. 当前 tag 清单（2026-10-18 整理结果）

| tag | 原分支 | 提交 | 说明 |
|---|---|---|---|
| `v2.0` | `main` | `2ff7a27` | **当前主线**：v2.0 工程化（tests / tools/config_ui / run.py 拆分）+ v1.0 绘图修复 |
| `v1.0` | `LHAv1.0` | `548290b` | 论文实验版：HMMEM 主线 + MELD/IEMOCAP 全量实验 + 绘图修复 |
| `archive/hmmem2.0` | `HMMEM2.0` | `ccb2cb6` | HMMEM 2.0 |
| `archive/hama-dbm1.0` | `HAMA-DBMv1.0` | `5a8ec78` | HAMA-DBM 1.0 |
| `archive/hmmem` | `HMMEM` | `248562c` | HMMEM 训练框架初版 |
| `archive/hmmem1.1` | `HMMEM1.1` | `8b616ea` | HMMEM 1.1 三模态任意组合消融 |
| `archive/moe0.4` | `MoE0.4` | `7579596` | MoE 0.4 |
| `archive/moe0.3` | `MoE0.3` | `168873a` | MoE 0.3 |
| `archive/moe0.3.1` | `MoE0.3.1` | `dbdfe38` | MoE 0.3.1 |
| `archive/ver0.2` | `ver0.2` | `c0aca34` | 新增 iemocap4 / iemocap6 数据集支持 |
| `archive/ver0.1` | `ver0.1` | `5a338b2` | MSE-Adapter 统一框架初版 |

`v1.0` 与 `v2.0` 的差异只有 15 个文件：`v2.0` 多了 `tests/`、`tools/config_ui.py`、`utils/gpu_utils.py`、`config/cli_args.py`、`data/context_utils.py` 与 `run.py` 拆分；`v1.0` 的 4 个绘图/性能提交已全部 cherry-pick 进 `v2.0`。

## 6. 提交信息

沿用仓库既有风格，前缀 + 中文描述说明**为什么**改（diff 已经说明了改什么）：

```
feat:     新能力（新模型、新数据集、新实验流程）
fix:      修 bug / 修评测、绘图、配置解析的异常
update:   配置、超参、日志、实验设置的调整
train:    仅新增训练记录（logs/ 已不进库，此类提交应显著减少）
refactor: 结构重排但不改行为
test:     测试
chore:    仓库治理（.gitignore、文档、依赖、CI）
```

## 7. 产物治理

- `logs/`、`Fig/`、`*.log`、`runs/`、`outputs/` 已加入 `.gitignore` 并移出版本库跟踪（历史中仍可追溯，例如 `git show v1.0:logs/hmmem-meld-qwen.log`）。
- 训练日志、绘图结果、checkpoint 不放版本库：本地保留 + 网盘/实验平台归档；需要长期留存的图表随论文版本走 GitHub Release 的 assets。
- 提交前用 `git status` 检查，只提交代码与配置；需要入库的少数图表请显式 `git add -f`。

## 8. 版本分发

对外（审稿人、合作者、开源用户）**发 tag + GitHub Release，不让人「去某个分支找代码」**：

```powershell
git tag -a v2.0 -m "LHAF v2.0"
git push origin v2.0
# 然后在 GitHub 上对该 tag 建 Release，附 zip 与实验说明
```

## 9. 恢复与回滚

- 整理前的全量备份：`../LHAF-all-20261018.bundle`（含 13 条分支、全部提交，`git bundle verify` 通过）。
  ```powershell
  git clone ../LHAF-all-20261018.bundle restored-lhaf
  ```
- 需要基于旧版本继续干活：从 tag 拉临时分支，而不是恢复旧分支名。
  ```powershell
  git switch -c hotfix/v1.0-plot v1.0
  ```
- 误推 tag：`git push origin :refs/tags/<name>` 后本地 `git tag -d <name>`。
- `main` 误提交：`git revert <sha>`（`main` 不做 rebase / 不强推）。
