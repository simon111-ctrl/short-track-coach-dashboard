# 模型可靠性说明

本次所有 gender + distance 数据组的清洗后样本量均超过 500，未触发样本量不足标记。
分类任务应优先参考 balanced_accuracy、macro_f1 和 roc_auc。多分类 max_round、key_lap 通常比二分类任务更难，应避免强解释。
style_cluster 和 risk_detection 为无监督模型，不能按准确率解释，只能作为风格参考和风险提示。
网页端必须按 gender + distance 选择模型，不得混用男女标准。
