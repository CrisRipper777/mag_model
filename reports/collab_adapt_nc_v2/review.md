# Collab adapt NC v2 review

本文件记录实现者自审，不是独立审查。

- 自审状态：PASS；问题数 0。
- 重建 train/validation argmax 指标与混淆矩阵：1728 份 head/partition。
- 完整成功 attempt：216；attempt状态：`{"R_calibration_success": 36, "R_confirmation_success": 18, "S_calibration_success": 108, "S_confirmation_success": 54}`。
- test head forward / test metrics / real LP: 0。预测文件键必须精确为 train/validation IDs、availability 与各任务头 argmax；不含 logits/features/embedding/test。
- R/S选择锁按校准结果重算；confirmation 的LR/WD/config/seed与已提交锁逐格匹配。
- optimizer step只做一次global clip；诊断post-clip norm由裁剪后的.grad直接计算。
- 缓存仅保留当前实际输入/边/mask对象，核验版本与设备；输入requires_grad、inference tensor或身份变化时重建/禁用。
- 初值经通用按名Xavier后显式恢复gate/NULL；load_state_dict只恢复checkpoint值、不调用初值恢复。
- V/F关闭配置不会构造其专属模块；V初始函数等价，F增量锚定。
- v1模型、脚本、协议、报告与raw evidence没有工作树改动；用户已有CLAUDE删除和三个设计文档保持未暂存/未编辑。

## 测试记录

- `conda run -n yhf_env python -m pytest tests/test_collab_adapt.py -q`：见 test_record.json。
- `CUDA_VISIBLE_DEVICES=1 conda run -n yhf_env python -m pytest tests/test_collab_adapt.py -q`：GPU合成功能检查；见 test_record.json。
- 三图 R-M0/S5 smoke 各三步，train only，见 smoke_summary.json。

## 审计问题

- 无。

## Audit precision reconciliation

- The locked auditor first reported 1718 accuracy-only differences because it compared float32-serialized accuracy to float64 reconstruction at 1e-10.
- A separate post-run audit entry point used a 1e-7 tolerance for serialized accuracy. Integer counts, confusion matrices, and Macro-F1 remain exact checks.
- Corrected self-audit: PASS; train/validation prediction checks: 1728. See `audit_precision_correction.json`.
- This reconciliation changed no training source, protocol lock, selection, model state, or raw result.
