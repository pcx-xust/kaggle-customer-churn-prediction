# Predict Customer Churn EDA Report

## 1. EDA阶段结论
从目标变量分布来看，训练集中共有594194条样本，其中流失客户133817条，占比22.52%，未流失客户460377条，占比77.48%。该任务存在一定类别不平衡，因此后续模型评价不应仅依赖Accuracy，还应结合AUC、Recall、F1等指标综合判断模型效果。

数值变量分析显示，流失客户的平均tenure为17.13，明显低于未流失客户的42.23，说明使用时长较短的客户更容易流失。流失客户的MonthlyCharges均值为81.60，高于未流失客户的61.29，表明较高月费用可能提高客户流失风险。TotalCharges在流失客户中更低，但该现象主要由tenure较短导致，不能简单解释为低累计消费客户更容易流失。

类别变量分析显示，Month-to-month合约、Electronic check支付方式、Fiber optic互联网服务、无OnlineSecurity、无TechSupport以及SeniorCitizen=1的客户流失率明显较高。其中Electronic check客户流失率达到48.91%，Month-to-month客户流失率达到42.05%，Fiber optic客户流失率达到41.54%。相反，Two year合约客户流失率仅为1.00%，无互联网服务客户流失率为1.43%，说明长期合约和低服务绑定客户的流失风险较低。

综合来看，客户流失与合约类型、使用时长、支付方式、互联网服务类型、月费用以及增值服务配置密切相关。后续建模应重点关注Contract、tenure、PaymentMethod、InternetService、MonthlyCharges、OnlineSecurity、TechSupport等变量。
