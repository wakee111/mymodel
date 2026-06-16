# CLAUDE.md

## 协作与判断准则

- 允许明确说“不知道”或“当前证据不足”。绝对不要不懂装懂，也不要为了给出结论而忽略不确定性。
- 所有实验建议、论文表述建议和模型选择建议都必须有可检查的依据：来自代码实现、日志结果、验证集规则、公开论文中的理论，或明确标注为经验性判断。
- 不要讨好用户。优先给出真实、可复现、能经得起审稿追问的判断；如果用户的设想存在风险，需要直接指出。
- 不要用测试集结果反向制定实验规则。任何超参数、训练策略或模型变体的选择，都应表述为基于验证集或预先定义准则完成。
- 避免 cherry-picking。除非论文明确说明，否则不要按预测长度单独选择不同配置；更稳妥的做法是按数据集级别或任务级别统一选择。

## `freq_loss_alpha` 的使用准则

`freq_loss_alpha` 可以作为超参数使用。当前实现中的训练目标为：

```text
Loss = alpha * MSE + (1 - alpha) * FreqLoss
```

其中 `alpha = 1.0` 等价于纯 MSE，`alpha < 1.0` 表示引入频域损失。频域损失的理论动机是：时间序列预测不仅需要点值误差小，也可能需要频谱结构、周期成分和振荡模式更接近真实序列。它可以作为 MSE 的补充约束，但不保证在所有数据集和所有指标上都优于纯 MSE。

论文中建议将 `freq_loss_alpha` 定义为**数据集级超参数**，并通过验证集选择。根据当前日志结果，可以采用如下准则：

- Solar：使用 `freq_loss_alpha = 1.0`，即纯 MSE。现有 Solar 结果显示 FreDF 在 MAE/SMAPE 上有收益，但在主指标 MSE 上不稳定，3-stage 整体模型的平均 MSE 略差于纯 MSE。
- ETT 系列：可以使用 `freq_loss_alpha = 0.8`。现有 ETT 日志中，FreDF 对 ETTm1、ETTm2、ETTh2 基本呈正向收益，ETTh1 的 Freq1 单阶段结果也优于纯 MSE。
- 不建议按 `pred_len=96/192/336/720` 分别选择不同 alpha，除非论文专门设计并报告“按 horizon 调参”的验证流程。

推荐论文表述：

```text
We treat the frequency loss weight alpha as a dataset-level hyperparameter and select it on the validation set. For Solar, alpha=1.0 is selected, reducing the objective to pure MSE. For the ETT datasets, alpha=0.8 provides better validation performance and is used in the final model.
```

如果展示消融实验，建议正文至少展示 Solar 和一个 ETT 数据集。Solar 用来说明 FreDF 与纯 MSE 在不同指标上的 trade-off；ETT 用来说明频域损失在具有明显周期结构的数据集上可以带来收益。其余 ETT 数据集可放在整体结果表中与其他模型对比，附录可补充完整消融。
