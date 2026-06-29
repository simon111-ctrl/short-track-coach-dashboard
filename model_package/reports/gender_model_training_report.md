# 男女分组短道速滑模型重新训练报告

## 1. 旧项目结构总结

旧项目位于 `C:\Users\Administrator\Documents\short_track_model_training_20260628`。核心脚本为 `train_short_track_models.py`，旧模型包位于 `model_package`，原始数据位于 `data_raw`。旧项目已经按照 500m、1000m、1500m 三个距离分别建模，并为每个距离训练 advancement、grade、max_round、final_entry、tactical_style、key_lap、style_cluster、risk_detection 八类模型。

## 2. 旧版清洗规则总结

旧版规则要求每条样本必须具有该距离完整的每圈成绩和每圈位置。任意圈速或圈位缺失的行会被删除。总成绩缺失但圈速完整时，用每圈成绩之和重建总成绩，并保留缺失标记。qual 只用于构造晋级标签，不作为预测输入。运动员姓名、样本 ID、轮次、source_url 等只用于审计、分组验证或标签构造，不作为模型特征。

## 3. 旧版特征工程规则总结

旧版特征包括每圈成绩、每圈位置、重建总成绩、平均圈速、圈速波动、前中后段节奏、起终段比例、位置均值和波动、总位置增益、超越次数、位置损失次数，以及 500m、1000m、1500m 各自的专项节奏和位置控制特征。grade 模型为避免泄漏，不使用总成绩、重建总成绩、平均圈速和原始绝对圈速。

## 4. 旧版模型训练方式总结

分类任务沿用 RandomForestClassifier，预处理为中位数缺失值填充加 StandardScaler。style_cluster 沿用 KMeans。risk_detection 沿用 IsolationForest。验证方式包括运动员分组留出、赛季时间外推留出、随机分层留出。旧项目是分类、聚类和风险模型体系，不是成绩时间回归模型，因此本次评估继续使用 accuracy、balanced_accuracy、macro_f1、weighted_f1、roc_auc、silhouette、anomaly_rate 等旧版指标。

## 5. 新版 gender 数据检查结果

- 500m：原始 12756 行，完整圈速/圈位清洗后 10116 行，其中 male 5838 行，female 4278 行。
- 1000m：原始 12704 行，完整圈速/圈位清洗后 9996 行，其中 male 5704 行，female 4292 行。
- 1500m：原始 11624 行，完整圈速/圈位清洗后 9013 行，其中 male 5228 行，female 3785 行。

gender 原始值为 Man 和 Woman，已标准化为 male 和 female。未发现缺失或无法识别的 gender 值。

## 6. gender 字段标准化规则

man、men、male、m、男、男子 统一为 male；woman、women、female、f、女、女子 统一为 female。gender 仅用于拆分训练数据和网页端选择模型，不进入任何模型特征。

## 7. 男女样本量

- female 总样本量：12355
- male 总样本量：16770

## 8. 每个距离下 male 和 female 样本量

- female 1000m：4292 行
- male 1000m：5704 行
- female 1500m：3785 行
- male 1500m：5228 行
- female 500m：4278 行
- male 500m：5838 行

## 9. 每个模型训练结果

- female_1000m_advancement：训练样本 3896，特征 70，accuracy=0.8526814440703202, balanced_accuracy=0.8551956326429394, macro_f1=0.852007078636694, roc_auc=0.9316469718278108，样本量充足。
- male_1000m_advancement：训练样本 5308，特征 70，accuracy=0.8345481089746688, balanced_accuracy=0.8350468973478969, macro_f1=0.8342942800505943, roc_auc=0.919289065820624，样本量充足。
- female_1000m_final_entry：训练样本 4292，特征 70，accuracy=0.7693129930816812, balanced_accuracy=0.7495136139806028, macro_f1=0.7455529966395141, roc_auc=0.8384523906257143，样本量充足。
- male_1000m_final_entry：训练样本 5704，特征 70，accuracy=0.7698004898480663, balanced_accuracy=0.6923946143626015, macro_f1=0.6926928454291402, roc_auc=0.8062118975052405，样本量充足。
- female_1000m_grade：训练样本 4292，特征 45，accuracy=0.585942644489598, balanced_accuracy=0.587553471366319, macro_f1=0.5836404862478194, roc_auc=nan，样本量充足。
- male_1000m_grade：训练样本 5704，特征 45，accuracy=0.5611274256639364, balanced_accuracy=0.559951533667976, macro_f1=0.5617488180398164, roc_auc=nan，样本量充足。
- female_1000m_key_lap：训练样本 4292，特征 70，accuracy=0.6884473911155752, balanced_accuracy=0.6918644767421872, macro_f1=0.6774105859264689, roc_auc=nan，样本量充足。
- male_1000m_key_lap：训练样本 5704，特征 70，accuracy=0.7126161256959761, balanced_accuracy=0.7234475719511785, macro_f1=0.7064065689151587, roc_auc=nan，样本量充足。
- female_1000m_max_round：训练样本 4292，特征 70，accuracy=0.4999555968211422, balanced_accuracy=0.3165285187500259, macro_f1=0.2762464075062891, roc_auc=nan，样本量充足。
- male_1000m_max_round：训练样本 5704，特征 70，accuracy=0.4321503479429043, balanced_accuracy=0.3044920085518016, macro_f1=0.2867312053214917, roc_auc=nan，样本量充足。
- female_1000m_risk_detection：训练样本 4292，特征 15，anomaly_rate=0.1199906803355079, risk_score_mean=-0.0500962582909694，样本量充足。
- male_1000m_risk_detection：训练样本 5704，特征 15，anomaly_rate=0.1200911640953716, risk_score_mean=-0.0540217786299239，样本量充足。
- female_1000m_style_cluster：训练样本 4292，特征 18，silhouette=0.2403499226571086，样本量充足。
- male_1000m_style_cluster：训练样本 5704，特征 18，silhouette=0.2473938554094191，样本量充足。
- female_1000m_tactical_style：训练样本 4292，特征 45，accuracy=0.9926980444227448, balanced_accuracy=0.990365537710776, macro_f1=0.9906819159630476, roc_auc=nan，样本量充足。
- male_1000m_tactical_style：训练样本 5704，特征 45，accuracy=0.994600579937812, balanced_accuracy=0.9932395020442426, macro_f1=0.9937900411165032, roc_auc=nan，样本量充足。
- female_1500m_advancement：训练样本 3168，特征 92，accuracy=0.8595825229606362, balanced_accuracy=0.8588885195099086, macro_f1=0.8591467647514944, roc_auc=0.9347003668717022，样本量充足。
- male_1500m_advancement：训练样本 4629，特征 92，accuracy=0.8658993841384346, balanced_accuracy=0.8650974268949806, macro_f1=0.8645238510395764, roc_auc=0.9337387737344864，样本量充足。
- female_1500m_final_entry：训练样本 3785，特征 92，accuracy=0.7568157668833595, balanced_accuracy=0.7568362030650925, macro_f1=0.7518218371924733, roc_auc=0.8392559143892641，样本量充足。
- male_1500m_final_entry：训练样本 5228，特征 92，accuracy=0.7781173903536756, balanced_accuracy=0.7569507569724568, macro_f1=0.7485915667445119, roc_auc=0.8508808670641933，样本量充足。
- female_1500m_grade：训练样本 3785，特征 61，accuracy=0.689791887861074, balanced_accuracy=0.6904140466431197, macro_f1=0.69355889074354, roc_auc=nan，样本量充足。
- male_1500m_grade：训练样本 5228，特征 61，accuracy=0.6502157212641354, balanced_accuracy=0.6493919578631697, macro_f1=0.6478513863648458, roc_auc=nan，样本量充足。
- female_1500m_key_lap：训练样本 3785，特征 92，accuracy=0.6255089068556156, balanced_accuracy=0.6398459193265142, macro_f1=0.6197768564087135, roc_auc=nan，样本量充足。
- male_1500m_key_lap：训练样本 5228，特征 92，accuracy=0.6255521177319738, balanced_accuracy=0.6495402270737851, macro_f1=0.6246706654063349, roc_auc=nan，样本量充足。
- female_1500m_max_round：训练样本 3785，特征 92，accuracy=0.5910395684928123, balanced_accuracy=0.4205001094542127, macro_f1=0.3510704066856623, roc_auc=nan，样本量充足。
- male_1500m_max_round：训练样本 5228，特征 92，accuracy=0.5466342036480164, balanced_accuracy=0.355553736507385, macro_f1=0.2906963442444895, roc_auc=nan，样本量充足。
- female_1500m_risk_detection：训练样本 3785，特征 15，anomaly_rate=0.1202113606340819, risk_score_mean=-0.0546826415073613，样本量充足。
- male_1500m_risk_detection：训练样本 5228，特征 15，anomaly_rate=0.1201224177505738, risk_score_mean=-0.0531473791426517，样本量充足。
- female_1500m_style_cluster：训练样本 3785，特征 18，silhouette=0.1742441819831002，样本量充足。
- male_1500m_style_cluster：训练样本 5228，特征 18，silhouette=0.1967581609777091，样本量充足。
- female_1500m_tactical_style：训练样本 3785，特征 61，accuracy=0.9881748000134708, balanced_accuracy=0.9813957499907026, macro_f1=0.973358463501344, roc_auc=nan，样本量充足。
- male_1500m_tactical_style：训练样本 5228，特征 61，accuracy=0.9877349350444956, balanced_accuracy=0.9785275558888304, macro_f1=0.9761874377563028, roc_auc=nan，样本量充足。
- female_500m_advancement：训练样本 3896，特征 55，accuracy=0.8461901060281685, balanced_accuracy=0.8473081266508298, macro_f1=0.8448768576868771, roc_auc=0.928597175381662，样本量充足。
- male_500m_advancement：训练样本 5460，特征 55，accuracy=0.8608723566575368, balanced_accuracy=0.8614353274750299, macro_f1=0.8608018868997154, roc_auc=0.940221593656172，样本量充足。
- female_500m_final_entry：训练样本 4278，特征 55，accuracy=0.7725729411832956, balanced_accuracy=0.7485958149377442, macro_f1=0.7441009568231048, roc_auc=0.8412105781668431，样本量充足。
- male_500m_final_entry：训练样本 5838，特征 55，accuracy=0.7721071386660697, balanced_accuracy=0.7089005825421971, macro_f1=0.6988174593317765, roc_auc=0.806908208370892，样本量充足。
- female_500m_grade：训练样本 4278，特征 34，accuracy=0.5119444077601454, balanced_accuracy=0.5053202765623711, macro_f1=0.50856629881445, roc_auc=nan，样本量充足。
- male_500m_grade：训练样本 5838，特征 34，accuracy=0.5426927093931542, balanced_accuracy=0.5433036986292539, macro_f1=0.5419425472866843, roc_auc=nan，样本量充足。
- female_500m_key_lap：训练样本 4278，特征 55，accuracy=0.8995126960093285, balanced_accuracy=0.8740966452868593, macro_f1=0.8702785520230675, roc_auc=nan，样本量充足。
- male_500m_key_lap：训练样本 5838，特征 55，accuracy=0.8997897799655723, balanced_accuracy=0.8887591311809669, macro_f1=0.8820119071866865, roc_auc=nan，样本量充足。
- female_500m_max_round：训练样本 4278，特征 55，accuracy=0.4714512888815179, balanced_accuracy=0.303779487590836, macro_f1=0.2659364074393297, roc_auc=nan，样本量充足。
- male_500m_max_round：训练样本 5838，特征 55，accuracy=0.386386400850998, balanced_accuracy=0.2859438442757512, macro_f1=0.2551994158890482, roc_auc=nan，样本量充足。
- female_500m_risk_detection：训练样本 4278，特征 15，anomaly_rate=0.1201496026180458, risk_score_mean=-0.0513361256843168，样本量充足。
- male_500m_risk_detection：训练样本 5838，特征 15，anomaly_rate=0.1200753682768071, risk_score_mean=-0.0620382724872385，样本量充足。
- female_500m_style_cluster：训练样本 4278，特征 18，silhouette=0.4499170325595912，样本量充足。
- male_500m_style_cluster：训练样本 5838，特征 18，silhouette=0.388514647634264，样本量充足。
- female_500m_tactical_style：训练样本 4278，特征 34，accuracy=1.0, balanced_accuracy=1.0, macro_f1=1.0, roc_auc=nan，样本量充足。
- male_500m_tactical_style：训练样本 5838，特征 34，accuracy=1.0, balanced_accuracy=1.0, macro_f1=1.0, roc_auc=nan，样本量充足。

## 10. 特征重要性

每个模型的置换重要性或无监督解释结果已保存到 `reports/explanation_<model_id>.csv`，总表保存到 `reports/model_explanations.csv`。网页和人工分析应优先查看对应 gender + distance + task 的解释文件，不要跨性别共用解释。

## 11. 可靠性说明

本次清洗后每个 gender + distance 的样本量均为数千行，按样本量看没有明显不足。分类模型仍需结合各任务的 balanced_accuracy、macro_f1 和 roc_auc 判断可靠性；max_round 等多分类任务天然更难，解释时应比 advancement、grade、final_entry 更谨慎。style_cluster 和 risk_detection 是无监督模型，没有准确率或 F1，不能用分类指标解释。

## 12. 最终保存路径

所有新版输出均保存于 `C:\Users\Administrator\Documents\short_track_model_training_20260628\gender_retrain_outputs`。模型位于 `models/<gender>/<distance>/<task>/model.joblib`，特征列位于同目录 `features.json`，网页清单位于 `web/web_model_manifest.json`。

## 13. 网页端调用方式

网页端必须先将用户选择的性别标准化为 male 或 female，再结合距离 500m、1000m、1500m 和任务名选择模型。例如用户选择“男”和“500m”的 grade 任务，应调用 `male_500m_grade`。不要加载统一模型，也不要把 gender 作为普通特征传入模型。

## 14. 接入网页时需要注意的字段

网页端需要提供对应距离的完整每圈成绩和每圈位置。500m 需要 5 圈，1000m 需要 9 圈，1500m 需要 14 圈。运动员姓名、qual、轮次、source_url 可以用于展示或审计，但不能作为模型输入特征。

## 15. 预测结果解释

分类模型输出预测标签和概率。grade 中 3 表示同性别同距离内最快四分位，0 表示最慢四分位。advancement、final_entry 表示对应事件概率。style_cluster 表示同性别同距离内部的节奏/位置控制聚类。risk_detection 输出异常或风险倾向，需要结合训练数据分布和教练判断。

## 16. 本次训练限制

本次没有新增成绩时间回归模型，因此不输出 MAE、RMSE、R²。所有结论基于旧项目的分类、聚类和风险建模体系。gender 来源中存在 high 和 medium 置信度，后续如需更严格版本，可只用 high 置信度数据再训练对照模型。
