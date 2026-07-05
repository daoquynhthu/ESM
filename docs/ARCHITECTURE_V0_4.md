# Elastic Sparse Machine 架构规划书

**版本:** Draft 0.4 / 原则一致性修正版  
**日期:** 2026-07-04  
**定位:** 后 SBM 路线的独立跃迁架构设计草案  
**核心约束:** 稀疏、在线学习、CPU-first、严格因果、无外部分配时间尺度、无先验生物模块堆叠、可归因审计  
**本版修订重点:** 在 Draft 0.3 的可实现规格基础上，移除研究报告中不应进入实现的三类占位符：隐藏线性加权 confidence、固定 token/occurrence 外部窗口、与 E-1A 风险不一致的固定工期估算；并把 claim / rescue / genesis / roadmap 改成相对结构预算和 gate-driven 退出规则。

---

## 0. 摘要

Elastic Sparse Machine，简称 **ESM**，是对 Sparse Branch Machine 后续路线的一次架构跃迁设计。

当前 SBM 的负结果不应解释为“稀疏智能路线失败”，而应解释为一个更窄的结论：

> 以地址程序、局部 token residual、内容递归和结构生命周期为核心的稀疏地址机，尚不足以成为竞争性语言模型主干。

ESM 保留 SBM 中真正有价值的部分：持久结构、稀疏活动、局部更新、结构生命周期、严格因果训练、CPU 友好存储与执行；但不再把“地址程序”当作智能的核心单元。

ESM 的核心转向是：

> 智能结构之间的差异，不由外部指定的“短期/长期模块”或“快慢时间尺度”定义，而由结构自身的 **改变容易程度** 定义。

也就是说，系统不预设短期记忆、长期记忆、海马、皮层、基底节、小脑等模块。系统只有一个统一的稀疏结构场。结构可以变得容易改变，也可以变得难以改变；可以被覆盖，可以分叉，可以冻结，可以退役。所谓时间尺度，是结构阻抗的涌现结果，而不是工程师预先指定的外部超参。

但是，本版必须明确收缩 ESM 的贡献边界：ESM 所在的机制空间并非空白。它与 HTM 的 sparse distributed representation、Temporal Memory、active dendrite / segment sequence memory，以及 ART 的 stability-plasticity dilemma、vigilance、category proliferation 问题高度相邻。ESM 的新意不应表述为“首次提出稀疏在线 segment 学习”，而应表述为：

> ESM 是一个 CPU-first、prequential、无 target leakage、成本审计、router 可审计、具有 genesis / fork / composition-depth 机制的 elastic sparse substrate，用于检验局部可塑稀疏结构能否在固定 active compute 下自组织出有预测价值的表征。

本版新增一个前置阶段：**Gate E-1**。在实现原始 Gate E0 之前，必须先验证：

1. sparse encoder 是否提供了超越 token/hash control 的表征；
2. eligibility / causal ledger 是否能解决延迟证据的信用分配；
3. genesis 是否能在无父结构时冷启动新结构；
4. router 是否没有掩盖 substrate 的真实能力；
5. segment / feature 是否具备有限组合深度，而不是停留在单层阈值检测器。

如果 Gate E-1 不通过，ESM 不应进入语言流实验。

Draft 0.3 进一步明确：Gate E-1A 不是普通工程 gate，而是整条路线的主要科学风险。E-1B/E-1D/E-1E 更接近机制正确性与预算正确性问题；E-1A 要求在没有 dense embedding、没有 global backprop、没有大规模离线共现预处理的条件下，在线形成具有 latent role 信息的 sparse representation。该点失败的先验概率高于其他 gate，因此必须拥有独立探索预算和更严格的 stop rule。

Draft 0.4 进一步规定：研究报告或实现草稿中的连续标量加权、固定 token 窗口、固定每 N token 结构增长率、固定工程周数，都不得自动成为架构规格。ESM 的规范层只允许三类常量：硬件预算常量、结构相对预算常量、离散 gate 阈值。凡是看起来像“0.5/0.3/0.2”线性经验权重、“最近 64 次”滑动窗口、“每 1000 token 最多 N 个 genesis”或“Phase A 固定两周”的内容，均视为待消融占位符，不作为默认算法。

---

## 1. 背景与问题重述

### 1.1 当前 SBM 的经验教训

SBM 已经证明了一些重要事实：

1. 持久逻辑节点与可移动物理存储是可行的。
2. 存储容量可以增长，而每步 active work 保持有界。
3. 局部结构可以通过 proposal、validation、ablation、accept、retire 等生命周期进行审计。
4. token cross-entropy、frozen evaluation、mmap corpus、checkpoint/resume、C/Python API 等实验工程可以稳定落地。
5. 稀疏结构确实可以带来局部预测增益。

但 SBM 同时暴露出几个根本问题：

1. 地址程序过于依赖 token identity、lag、matched successor 等表面结构。
2. 内容匹配和 follow 仍然是 token recurrence 级别的机制，不是真正的 latent relation。
3. 输出层修复、global prior、local residual tuning 往往贡献了主要 NLL 改善，容易掩盖结构学习本身是否有效。
4. 引入更多机制后，归因迅速变难。
5. 一旦加入多模块、多时间尺度、多套门控，系统会走向超参爆炸。

因此，新路线必须避免两个极端：

- 继续在旧地址程序框架中堆更复杂的指令；
- 按生物名词堆叠大量模块，形成另一个不可归因系统。

### 1.2 新路线的目标

ESM 的目标不是“模拟大脑结构”，而是抽取一个更基本的原则：

> 智能系统应当由大量稀疏活动结构组成，这些结构通过在线经验改变自身的可塑性、稳定性和分叉倾向，从而形成不同表观时间尺度和功能分化。

换句话说：

- 不设计短期记忆模块；设计高可塑性结构。
- 不设计长期记忆模块；设计高阻抗结构。
- 不设计海马模块；允许低阻抗新结构快速创生、分叉和试错。
- 不设计基底节模块；先使用显式可审计 scheduler，并把它标记为技术债，而不是把 router 当作涌现智能。
- 不设计小脑模块；让低成本 residual predictor 作为某类 element 的可能形态出现，而不是先验模块。

功能可以出现，但不能先验硬编码成生物模块。

---

## 2. 与已有研究的关系与差异

### 2.1 与 HTM / Temporal Memory 的关系

ESM 与 HTM 的相似性是直接的：

| HTM / Temporal Memory | ESM |
|---|---|
| Cell | Element |
| Dendritic segment | Segment |
| Synapse permanence | Synapse weight / confidence |
| Sparse Distributed Representation | Sparse active field / sparse encoder code |
| Predictive cells | Segment-matched / claim-issuing elements |
| Online sequence memory | Online prequential structural learning |
| Branching sequence prediction | fork / competing descendants / pending claims |

HTM sequence memory 的核心主张包括：在线连续学习 variable-order temporal sequences、多分支序列中保持多个预测、噪声鲁棒，以及使用稀疏分布式活动作为基础。ESM 必须承认这不是新机制空间。

ESM 与 HTM 的预期差异：

1. **严格 prequential evaluation**：预测先固定，再观察 target，所有 target-dependent 更新禁止影响当前预测。
2. **成本审计**：每个 element、segment、link、claim 都必须支付 storage / active compute / false-positive / interference 成本。
3. **genesis 与 fork 分离**：新结构冷启动与稳定结构冲突分叉不是同一事件。
4. **router 显式审计**：候选调度器不是智能的一部分，而是可替换、可 ablate、可报告的技术债。
5. **组合深度显式化**：segment match 结果必须能形成可引用 feature event，否则系统只是平坦阈值检测器。
6. **语言建模不从 token hash 开始**：必须把 sparse representation formation 作为 Gate E-1A，而不是假设 token 派生 signature 自然足够。

### 2.2 与 ART / stability-plasticity dilemma 的关系

ART 的核心问题是 stability-plasticity dilemma：系统既要学习新知识，又不能破坏已有知识。ART 使用 bottom-up input 与 top-down template 的匹配，以及 vigilance threshold 决定是否接受已有类别或搜索/创建新类别。

ESM 的 `fork_pressure`、`resistance`、`plasticity`、`genesis` 与 ART 的 vigilance / category formation 高度相关。

因此，ESM 必须吸收 ART 的直接教训：

1. **category proliferation** 对应 ESM 的 fork/genesis explosion。
2. vigilance / threshold 不当会造成过度细分或过度合并。
3. 训练顺序会影响类别形成；ESM 的 stream order 必须作为实验变量报告。
4. threshold 不能只是手写常数，必须由 codelength gain、interference、coverage、structural rent 共同约束。

ESM 不应把 `fork_pressure_threshold` 写成一个孤立超参，而应把 fork/genesis 纳入共同的结构账本：

```text
surprise_reduction
- description_cost
- storage_cost
- active_cost
- interference_cost
- false_positive_cost
> 0
```

只有在净收益为正时，结构增长才有资格发生。

### 2.3 与 continual learning 的关系

ESM 处理的是在线学习与持续学习问题，但不能简单使用多模块 replay / fast-slow memory，因为本项目约束禁止外部分配时间尺度。

ESM 的处理方式：

- 不设置 explicit short-term / long-term module；
- 不设置固定 replay interval；
- 不设置固定 memory decay window；
- 用 plasticity / resistance / claim rent / structural cost 产生表观快慢差异；
- 用 causal ledger 与 pending claim 处理延迟信用，而不是固定时间衰减 trace。

### 2.4 与 threshold circuit / 深度表达力的关系

单个 Segment 本质上是一个稀疏线性阈值检测器：

```text
sum_i w_i * active_i >= threshold
```

单层阈值单元不能表达任意深层组合逻辑。要获得组合表达效率，系统需要某种 feature-of-feature 机制：Segment 的 match 结果必须能够被后续 Segment 引用。

因此，ESM 必须显式支持 bounded composition rounds：

```text
Round 0: sparse encoder bits
Round 1: segment-derived feature events
Round 2: feature-of-feature events
Round D: prediction-active elements
```

这里的 D 是 CPU 计算深度预算，不是记忆时间尺度。第一版不得调 D 作为性能超参，只能用 D=1/2/3 做表达力 gate。

### 2.5 与 Transformer scaling law 的关系

ESM 不应宣称“逃离幂律”或“非幂律智能”这种不可证伪口号。

更准确的假设是：

> 在 fixed active compute、online stream、persistent growing memory 的制度下，ESM 试图获得更高的 predictive gain per active operation / per stored byte，而不是在 dense pretraining scaling-law 制度下正面击败 Transformer。

因此，原文中的“非幂律”应替换为可测不等式：

```text
Given fixed active budget A and growing stored capacity M:

active_cost_per_step(M, A) ≈ O(A)
ΔNLL(M, A) = NLL(control) - NLL(ESM) > 0
storage_efficiency = ΔNLL / stored_bytes
compute_efficiency = ΔNLL / active_ops
```

如果 ΔNLL 只能通过 active compute 线性增长获得，则 ESM 的 CPU-first 稀疏路线失败。

---

## 3. 设计原则

### 3.1 稀疏性是活动原则，不是查表原则

稀疏智能不是“用稀疏哈希表替代神经网络”。

当前 SBM 的最大局限在于，它很容易退化成：

```text
token/history -> address/signature -> local node -> output residual
```

这是一种强工程化的稀疏查表范式。它可以高效，可以审计，但很难长出真正的抽象表征。

ESM 的稀疏性应当表达为：

```text
input event -> sparse active field -> local competition -> predictive state -> structural adaptation
```

结构被激活不是因为某个地址完全命中，而是因为多个稀疏证据源共同竞争后，它在当前上下文中最适合参与预测。

### 3.2 不引入外部分配时间尺度

系统不得预设：

```text
fast memory
slow memory
episodic memory
semantic memory
short-term buffer
long-term consolidation interval
```

所有表观时间尺度必须来自结构自身状态：

```text
effective_timescale(element) ~= 1 / plasticity(element)
```

一个结构长期稳定，不是因为它属于长期模块，而是因为它积累了高 utility、高 reliability、低 interference，从而降低了 plasticity、提高了 resistance。

一个结构短期快速适应，不是因为它属于短期模块，而是因为它低 resistance、高 plasticity，允许快速改写或被淘汰。

### 3.3 不先验堆叠生物模块

ESM 初始架构不得拆成：

```text
Cortex
Hippocampus
BasalGanglia
Thalamus
Cerebellum
```

这些名字可以作为解释类比，但不能成为一开始的工程分层。否则必然引入：

- 多套学习率；
- 多套 decay；
- 多套 gate；
- 多个 buffer；
- 多个评价窗口；
- 模块间 credit assignment；
- 模块间超参搜索。

ESM 的初始对象应尽可能少：

```text
Element
Link
Segment
FeatureEvent
PendingClaim
```

其中 FeatureEvent 与 PendingClaim 是机制所需的最小审计对象，不是额外生物模块。

### 3.4 Fork before overwrite

在线学习最核心的问题是：新证据到来时，是覆盖旧结构，还是建立新结构？

ESM 的原则是：

> 当高价值、高阻抗结构遇到系统性冲突时，优先分叉，而不是覆盖。

形式上：

```text
stable parent + systematic conflict -> plastic child fork
```

父结构保存旧知识，子结构以更高 plasticity 适应新上下文。子结构必须在后续 prequential / held-out 证据中证明自身价值，否则退役或合并。

### 3.5 Genesis before fork is possible

Fork 不是结构增长的唯一入口。

当系统还没有任何合格 parent 时，必须允许 **genesis**：

```text
unexplained sparse code + high surprise + low coverage + positive expected value -> new probe element
```

Genesis 与 fork 的区别：

| 机制 | 前提 | 初始化 | 风险 |
|---|---|---|---|
| Genesis | 没有合格 parent | 从当前 sparse code / provenance sketch 创建 probe element | 冷启动爆炸、噪声记忆 |
| Fork | 有稳定 parent 且遇到系统性冲突 | 继承 parent 局部结构并提高 plasticity | category proliferation |

Genesis 必须有严格预算和结构租金，否则会退化成“每个新输入建一个节点”。

### 3.6 Router 是技术债，不是智能机制

候选生成与激活调度在第一版中必须存在，但它不是 ESM 的哲学成果。它是 CPU 预算下不得不引入的 scheduler。

因此文档必须明确：

> Router is an explicit scheduler, not an emergent intelligence mechanism.

Router 必须：

- 可替换；
- 可 ablate；
- 报告 source exposure；
- 报告 positive-utility orphan；
- 不允许用复杂手写加权掩盖 substrate 失败。

第一版不使用 8 项线性加权 scorer，而使用 quota + lexicographic scheduler。

### 3.7 CPU-first 是底层约束

ESM 不是“先设计智能机制，之后再优化 CPU”。CPU 友好必须从原语层面进入设计。

硬约束：

```text
每步 active element 数量有界
每步 candidate 数量有界
每步 link scan 有界
每步 segment check 有界
每步 composition round 有界
每步 ledger 回溯有界
不得扫描全体元素
不得引入 dense attention
不得依赖大规模 ANN 检索
不得引入 dense global backprop
热路径使用 SoA 存储
候选数组连续化
结构 ID 稳定，物理槽可移动
所有结构增长都要计入 bytes 和 active cost
```

如果一个机制在理论上稀疏，但实现上需要大量随机指针跳转、全局扫描、动态分配或近似最近邻大索引，它不符合 ESM 的第一版设计。

---

## 4. 总体架构

ESM 的基本循环如下：

```text
input event
  -> sparse encoder / signature formation
  -> bounded candidate generation
  -> bounded composition rounds
  -> sparse competition
  -> active element set
  -> prediction
  -> prediction fixed
  -> observe target / outcome
  -> causal ledger credit
  -> plasticity / resistance update
  -> genesis / fork / merge / retire / quarantine
```

数学抽象：

```text
c_t = Encode(x_t, context_sketch_{t-1})
f_t^{0} = c_t
for r in 1..D:
    f_t^{r} = Compose(f_t^{r-1}, S_{t-1}, budget_r)
C_t = Candidates(f_t^{0:D}, S_{t-1})
a_t = SelectK(C_t)
ŷ_t = Predict(a_t, f_t^{0:D})
Fix(ŷ_t, Ledger_t)
y_t = Observe()
e_t = LossGradientOrCodelengthDelta(ŷ_t, y_t)
Credit(Ledger_t, e_t)
Adapt(S_{t-1}) -> S_t
```

严格因果顺序：

```text
所有会影响当前预测的状态，必须来自 t 之前。
当前 target y_t 只能影响 t 之后的状态。
```

任何违反该顺序的实现都视为 target leakage。

---

## 5. Sparse Encoder / Signature Formation

### 5.1 为什么 encode 是前置 Gate

ESM 的 Segment 匹配表达力上限由 active code 决定。如果 active code 只是 token id、n-gram、lag、hash，那么 ESM 会退化为更复杂的 n-gram 机器。

因此 encode 不是实现细节，而是 Gate E-1A。

必须同时实现三种 encoder，用作对照：

```text
Encoder A: Raw token/hash control
Encoder B: Online sparse competitive encoder
Encoder C: Predictive sparse encoder
```

如果 B/C 不能在控制 token recurrence 后提供额外预测信息，则 ESM 的主路线停止。

### 5.2 Encoder A: Raw token/hash control

用于防止自欺。

输入：

```text
token id
local token pair
coarse position bucket
document boundary flag
recent-token sketch
```

输出：

```text
fixed sparse binary signature
```

该 encoder 不是正式路线，只作为 lower-bound control。

### 5.3 Encoder B: Online sparse competitive encoder

目标：从输入事件中形成非平凡稀疏分布式表征。

对象：

```cpp
struct EncoderColumn {
    ColumnId id;
    SparseFeatureRefs receptive_field;
    float threshold;
    float usage_ema;
    float novelty_ema;
    float plasticity;
    float resistance;
    float predictive_utility;
};
```

步骤：

```text
1. extract raw event features
2. compute overlap with bounded receptive fields
3. apply homeostatic usage correction
4. select TopK encoder columns
5. update only active / near-active columns after prediction is fixed
6. apply anti-correlation pressure between frequently co-active columns
```

要求：

- 不使用 dense embedding matrix；
- 不使用 global backprop；
- 不扫描所有 columns；
- receptive field 稀疏；
- usage balance 必须报告。

### 5.4 Encoder C: Predictive sparse encoder

在 B 的基础上，允许 encoder column 根据预测贡献调整 receptive field 与 resistance。

核心原则：

```text
如果某 column 在控制 token/hash 后仍提供预测增益，则降低 plasticity，提高 resistance。
如果某 column 经常 co-active 但没有独立贡献，则提高 plasticity 或退役。
如果某 input pattern 无法被已有 columns 覆盖，则触发 encoder-column genesis。
```

### 5.5 Encoder 诊断指标

Gate E-1A 必须报告：

```text
same_token_context_split:
  同一 token 在不同上下文中是否激活不同 code 子集

role_sharing:
  不同 token 在相似预测角色中是否共享 code 子集

code_entropy:
  active columns 是否被少数 token 垄断

usage_balance:
  column usage Gini / entropy

predictive_mutual_information:
  code 在控制 token id / ngram 后是否有额外预测信息

hash_control_delta:
  learned encoder 是否超过 raw hash encoder

encoder_ablation_delta:
  移除 encoder plasticity 后预测是否下降
```

### 5.6 Encoder B/C 的研究风险分级

Encoder B/C 不是普通实现细节，而是 ESM 最主要的科学赌注。

必须明确区分两类问题：

```text
E-1B / E-1D / E-1E:
  主要是机制与预算正确性问题。
  只要规格足够清楚，第一版大概率可以实现并通过 toy gate。

E-1A:
  是表征学习问题。
  要求系统在无 dense embedding、无 global backprop、无离线大规模共现矩阵的条件下，在线形成携带 latent role 信息的稀疏表示。
```

因此 Gate E-1A 必须有独立探索预算，不能被视为与其他 Pre-Gate 同等难度。

最低要求：

```text
1. Encoder A/B/C 必须在同一 runner、同一 active budget、同一输出 head 下对照。
2. Encoder B/C 若只在 raw NLL 上略优，但 controlled predictive MI 不显著，则判定为未通过。
3. Encoder B/C 若主要收益来自 token identity 或 local n-gram leakage，则判定为未通过。
4. 若 B/C 在 toy latent-role stream 上失败，不进入真实语言流。
5. 若 B/C 在真实语言流上失败，但 E-1B/E-1C/E-1D/E-1E 通过，则结论应写成“ESM substrate 可运行，但在线 sparse representation formation 未闭合”。
```

这条规则防止把 encoder 失败误诊为 segment、fork、router 或 CPU budget 问题。

---

## 6. 核心对象

### 6.1 Element

Element 是 ESM 的基本持久结构单元。

```cpp
struct Element {
    ElementId id;
    uint32_t physical_slot;

    // predictive state
    float activation;
    float responsibility;
    float excitability;
    OutputHead output_head;

    // elastic state
    float plasticity;
    float resistance;
    float utility;
    float reliability;
    float interference;
    float novelty_pressure;
    float fork_pressure;
    float genesis_credit;

    // structural state
    ElementPhase phase;     // Probe, Active, Frozen, Dormant, Quarantined, Retired
    ElementId parent;
    uint32_t lineage_depth;

    // sparse relations
    SegmentRange segments;
    LinkRange outgoing_links;

    // cost accounting
    uint32_t byte_cost;
    float active_cost_ema;
    float false_positive_cost;
    float structural_rent;
};
```

Element 不代表固定语义概念。它只是一个可参与预测、可被路由、可改变可塑性的稀疏结构。

### 6.2 Segment

Segment 是上下文检测器。

```cpp
struct Segment {
    SegmentId id;
    ElementId owner;
    SegmentKind kind;       // Input, Context, Feature, Claim, Link

    SynapseRange synapses;
    float threshold;
    float match_score;
    float precision_ema;
    float recall_proxy;
    float false_positive_cost;

    float plasticity;
    float resistance;
    float utility;
    SegmentPhase phase;

    // composition support
    bool emits_feature_event;
    FeatureId feature_id;
};
```

第一版 Segment 是稀疏阈值检测器，但必须允许其 match 结果生成 FeatureEvent，否则无法形成组合深度。

### 6.3 Link

Link 是稀疏转移 / 路由 / 上下文关系。

```cpp
struct Link {
    LinkId id;
    ElementId src;
    ElementId dst;
    LinkKind kind;

    float strength;
    float reliability;
    float utility;
    float interference;
    float eligibility;
    float structural_rent;
};
```

`eligibility` 不再是固定时间衰减 trace，而是由 causal ledger 更新的贡献状态。

### 6.4 FeatureEvent

FeatureEvent 是组合深度的关键新增对象。

```cpp
struct FeatureEvent {
    FeatureId id;
    FeatureSourceKind source;   // Encoder, Segment, Link, Genesis
    uint32_t source_id;
    float confidence;
    float utility;
    float cost;
};
```

FeatureEvent 可以被后续 Segment 引用。

这允许：

```text
input feature
  -> segment match
  -> feature event
  -> higher-order segment match
  -> composed feature
  -> prediction
```

### 6.5 PredictionLedger

PredictionLedger 固定当前预测的因果来源。

```cpp
struct PredictionLedger {
    StepId step;
    PredictionId pred;

    ElementId active_elements[MAX_ACTIVE];
    float responsibility[MAX_ACTIVE];

    SegmentId matched_segments[MAX_MATCHED_SEGMENTS];
    LinkId causal_links[MAX_CAUSAL_LINKS];
    FeatureId feature_events[MAX_FEATURE_EVENTS];
    ClaimId pending_claims[MAX_PENDING_CLAIMS];

    float prediction_logprob;
    uint32_t byte_cost;
};
```

它是 eligibility 与 target-leakage 防御的核心。

### 6.6 PendingClaim

PendingClaim 用于处理延迟证据，不依赖固定时间尺度。

```cpp
struct PendingClaim {
    ClaimId id;
    ElementId issuer;

    SparseKey condition_key;
    SparseKey expected_future_evidence;

    float confidence;
    float utility;
    float storage_cost;
    float false_alarm_cost;
    float verification_credit;

    ClaimPhase phase;       // Open, Verified, Failed, Retired
};
```

Claim 不按固定 step decay，而是支付 storage rent。长期无贡献的 claim 由于 rent 和 false_alarm_cost 自然退役。

---

## 7. Elasticity 状态

### 7.1 Plasticity

Plasticity 表示结构改变容易程度。

```text
plasticity high:
  参数容易被更新
  segment 容易改写
  link 容易改变
  output head 容易调整
  结构容易被合并或淘汰

plasticity low:
  参数更新幅度小
  segment 稳定
  link 稳定
  output head 稳定
  结构更倾向保留旧知识
```

核心更新形式：

```text
Δθ_i = plasticity_i * local_credit_i
```

Plasticity 不是固定学习率，而是结构状态变量。

### 7.2 Resistance

Resistance 表示结构抵抗改变的程度。

直观规则：

```text
utility 高且 reliability 高:
  resistance 上升
  plasticity 下降

interference 高:
  resistance 不应盲目上升
  可能触发 fork 或 quarantine

长期无贡献:
  resistance 下降
  plasticity 上升或 retire
```

不要把 resistance 设成外部年龄函数。年龄只能作为诊断，不作为主要规则。

### 7.3 Utility

Utility 使用 codelength 或 prediction gain 统一度量。

```text
utility_i ≈ loss_without_i - loss_with_i - structural_cost_i
```

第一版不必每步做完整反事实 ablation，但必须保存近似：

```text
responsibility-weighted loss delta
held-out probe delta
source-level ablation delta
rent-adjusted utility
```

### 7.4 Interference

Interference 衡量结构更新或激活对其他上下文造成的损害。

```text
interference_i increases if:
  structure helps context A but hurts context B
  update improves current step but degrades recent held-out probes
  segment fires in incompatible contexts
  claim repeatedly predicts wrong future evidence
```

高 interference 的结构不应简单删除。它可能意味着需要 fork。

### 7.5 Fork pressure

Fork pressure 应由以下因素产生：

```text
parent utility high
parent resistance high
current error systematic
current context separable
overwrite would damage old contexts
child expected gain > child cost
```

而不是由单个 threshold 决定。

### 7.6 Genesis pressure

Genesis pressure 独立于 fork pressure。

```text
coverage low
surprise high
no adequate parent
encoder code stable enough
structural budget available
expected gain > rent
```

Genesis 是冷启动和新颖模式学习的入口。

---

## 8. 学习规则

### 8.1 因果顺序

每一步必须遵守：

```text
1. encode input using only past state
2. generate candidates using only past state
3. select active set
4. produce prediction
5. freeze prediction and ledger
6. observe target
7. compute loss/error
8. apply credit through ledger
9. update structures for future steps
```

禁止：

```text
target 参与当前 routing
target 参与当前 encode
target 参与当前 segment match
target 参与当前 prediction mixture
target 改写当前 ledger
```

### 8.2 局部预测更新

当前 active element 的 output head 更新：

```text
local_credit_i = responsibility_i * error_signal
Δoutput_i = plasticity_i * local_credit_i
```

但是这只处理当前预测直接贡献，不处理跨步依赖。

### 8.3 Causal eligibility ledger

ESM 不采用固定 λ 的时间衰减作为第一原则。Eligibility 来自当前预测的因果账本，而不是来自“几步之前”这个外部时间概念。

每个 FeatureEvent、SegmentMatch、LinkActivation、ElementActivation 在 ledger 中都形成一个有向无环 provenance graph：

```text
encoder feature
  -> segment match
  -> feature event
  -> candidate element
  -> active element
  -> output contribution
```

第一版最小算法如下。

#### 8.3.1 path_responsibility 最小算法

每个 ledger node 记录：

```cpp
struct LedgerNode {
    LedgerNodeId id;
    LedgerNodeKind kind;
    float local_weight;          // match score, link strength, feature confidence, or output mixture weight
    float forward_mass;          // computed before target is observed
    SmallVec<LedgerNodeId> parents;
    SmallVec<LedgerNodeId> children;
};
```

前向质量传播：

```text
1. output nodes start with abs(output_contribution_i).
2. normalize output masses so sum(output_mass) = 1.
3. traverse ledger DAG backward from output nodes.
4. child mass is distributed to parents proportional to parent.local_weight.
5. cap ancestors per node by P to keep ledger bounded.
6. path_responsibility(z) = accumulated_mass[z].
```

伪代码：

```text
for output_node o:
    mass[o] = abs(o.output_contribution) / sum_abs_output_contrib

for node in reverse_topological_order_from_outputs:
    denom = sum(max(parent.local_weight, eps) for parent in node.parents)
    for parent in topP(node.parents):
        mass[parent] += mass[node] * max(parent.local_weight, eps) / denom

path_responsibility(z) = clamp(mass[z], 0, 1)
```

这里不允许任意图回溯。Ledger 只保存当前预测实际用到的 bounded provenance DAG。未进入 ledger 的结构不会获得当前 error 的 credit。

#### 8.3.2 causal_confidence 最小算法

`causal_confidence` 不得使用隐藏线性加权 scorer。禁止把多个诊断量写成类似 `0.5*a + 0.3*b + 0.2*c` 的经验公式，除非该公式已经被单独 gate 和 ablation 证明。第一版使用 **离散序数规则**。

每个结构先计算三个离散诊断：

```text
precision_bucket:
  High    if recent verified_gain_count / resolved_count >= high_precision_floor
  Low     if recent false_alarm_count dominates
  Unknown otherwise

resolution_bucket:
  High    if claims usually resolve before rent exhaustion
  Low     if claims often expire or remain unresolved while paying rent
  Unknown otherwise

stability_bucket:
  High    if support sign is consistent across distinct contexts
  Low     if support alternates sign or causes interference
  Unknown otherwise
```

然后按保守组合规则得到 confidence：

```text
if any bucket == Low:
    causal_confidence = Low
elif at least two buckets == High:
    causal_confidence = High
else:
    causal_confidence = Medium
```

离散 bucket 映射到数值只允许在最后一步用于 credit magnitude：

```text
Low    -> 0.25
Medium -> 0.50
High   -> 1.00
```

不同结构类型只影响各 bucket 的原始统计来源，而不引入新的线性加权：

```text
ElementActivation:
  precision source = verified prediction utility
  resolution source = active-path claim resolution
  stability source = sign consistency across contexts

SegmentMatch:
  precision source = match success rate above threshold
  resolution source = downstream claim resolution from this segment
  stability source = false-positive/interference rate

LinkActivation:
  precision source = destination utility after traversal
  resolution source = linked claim settlement
  stability source = link reuse without negative credit

FeatureEvent:
  precision source = promoted feature predictive utility
  resolution source = feature-backed claim settlement
  stability source = round-specific false positive rate

EncoderColumn:
  precision source = controlled predictive utility after token/hash controls
  resolution source = downstream claim settlement through this code
  stability source = usage balance and context split consistency

PendingClaim:
  precision source = verified vs failed claim counts
  resolution source = rent-adjusted resolution before eviction
  stability source = support across distinct issuers / contexts
```

如果某字段尚未有统计量，使用 `Unknown`，不得伪装成 High；日志中标记 `cold_confidence_default`。

#### 8.3.3 credit 归一化

```text
raw_credit(z) = error_t
              * path_responsibility(z)
              * causal_confidence(z)

credit(z) = raw_credit(z) / max(sum_abs_raw_credit, eps)
```

为了避免“信用凭空放大”，每步总 credit magnitude 必须被限制：

```text
sum_z abs(credit(z)) <= abs(error_budget_t)
```

路径来源包括：

```text
active element
matched segment
causal link
feature event
pending claim
encoder column
```

### 8.4 Pending claim credit

对于“若干步之后才验证”的结构，使用 PendingClaim。

Claim 分成两类信用：

```text
issue credit:
  claim 发放时保存 issuer、source_round、source_feature、issuing_segment、issue_responsibility。

verification credit:
  后续 evidence 匹配或违背 claim 时，把 verified_gain / false_alarm_cost 回写给 issuer path。
```

#### 8.4.1 Claim 发放条件

Claim 不是每个 active element 自动发放。第一版只允许以下条件之一触发：

```text
Template claim:
  matched ClaimSegment exists
  and issue_responsibility > issue_resp_floor
  and claim template has positive rent-adjusted utility
  and claims_issued_per_step quota available

High-value element claim:
  active element responsibility in top claim_issuer_quota
  and element.claim_utility_ema > 0
  and expected_future_evidence template exists

Probe claim:
  prediction_surprise high
  and coverage low or composition_gap high
  and probe_claims_per_step quota available
  and open_claims_total below cap
```

所有 claim 必须带有 `expected_future_evidence`。如果没有 evidence template，则只能创建 probe claim；probe claim 的 quota 必须远小于 template claim。

#### 8.4.2 Claim 验证

每步只检查有限个 claim：

```text
candidate_claims = claim_index.lookup(current_feature_events, max=C_hit)
checked_claims = top_by(issue_confidence, utility, rent_paid, max=C_check)
```

验证规则：

```text
verification_match_score = similarity(current_evidence, expected_future_evidence)

if verification_match_score >= verify_floor:
    phase = Verified
    verification_credit = verified_gain - rent_paid - storage_cost

if contradiction_score >= fail_floor:
    phase = Failed
    verification_credit = - false_alarm_cost - rent_paid

else:
    pay storage_rent
```

Claim 不靠固定过期时间衰减，而是支付 rent：

```text
claim_value = verified_gain - storage_rent - false_alarm_cost
```

如果 claim 长期不被验证，它会因为 rent 变成负 utility。Rent 是结构成本，不是外部时间尺度；每次 claim 参与维护或检查都必须付费。

### 8.5 Plasticity / Resistance 更新

更新原则：

```text
if rent-adjusted utility positive and reliability high:
    resistance += small
    plasticity -= small

if utility positive but interference high:
    fork_pressure += increase

if utility negative and no special role:
    plasticity += increase
    retire_pressure += increase

if structure repeatedly causes false positives:
    segment threshold/resistance adjusted or quarantined
```

避免引入多个外部时间尺度。这里的 “small / increase” 不应成为一堆超参；第一版应统一使用 rank-based 或 sign-based update：

```text
positive bucket -> +1 resistance quantum
negative bucket -> -1 resistance quantum
conflict bucket -> +1 fork pressure quantum
```

---

## 9. Genesis 与 Fork 生命周期

### 9.1 Genesis 动机

没有 genesis，系统冷启动会陷入鸡生蛋问题。

Fork 要求存在 parent，但在初始阶段大量输入没有任何合格 parent。ESM 必须允许从 unexplained sparse code 创建 probe element。

### 9.2 Genesis 条件

Genesis 的输入不是只有 Round 0 encoder code，而是任意 composition round 中未被覆盖的活跃模式。

```text
UnexplainedPattern(r):
  round r feature set / coactivation pattern
  has low coverage by existing elements or segments
  contributes to prediction_surprise or composition_gap
  is not already represented by a positive-utility structure
```

Genesis 条件：

```text
if exists UnexplainedPattern(r) for r in 0..D
and prediction_surprise > surprise_floor
and parent_status in {NoAdequateParent, WeakParent}
and active_genesis_budget_available
and expected_gain_minus_rent > 0:
    create Probe Element from pattern at round r
```

第一版不允许无限 genesis。每步 genesis quota 必须极低，并且全局 Probe 存量有上限。

### 9.2.1 Parent status：消除 Genesis/Fork 死区

Genesis 与 fork 不得由两套互不相干的阈值制造“两不管”地带。每个上下文必须先计算 `parent_status`：

```text
best_parent = argmax_parent coverage_utility(parent, current_pattern)

if best_parent.utility < parent_floor
or best_parent.coverage < coverage_floor:
    parent_status = NoAdequateParent

else if best_parent.utility < fork_utility_min
or best_parent.resistance < fork_resistance_min:
    parent_status = WeakParent

else if best_parent.interference rising
and current_context separable
and overwrite_cost > fork_cost:
    parent_status = StableConflictParent

else:
    parent_status = StableCompatibleParent
```

决策表：

```text
NoAdequateParent:
  genesis allowed
  fork disallowed

WeakParent:
  weak-parent rescue mode
  genesis allowed as SiblingProbe or RefinementProbe
  parent adaptation allowed
  weak parent is not allowed to veto genesis

StableConflictParent:
  fork allowed
  genesis only allowed if composition_gap remains after fork candidate construction

StableCompatibleParent:
  no genesis/fork by default
  ordinary adaptation only
```

这样不存在“parent 太弱无法 fork、但又强到阻止 genesis”的死区。WeakParent 不能阻止新结构创生。

### 9.3 Genesis 初始化

```text
phase = Probe
plasticity = high
resistance = low
utility = unknown
structural_rent = positive
source_round = r
source_pattern = UnexplainedPattern(r)
segments = sparse pattern sketch from round r
output_head = weak residual
lineage = root | weak_parent_refinement | composition_probe
```

Probe element 必须尽快证明自己有 utility。否则退役。

Genesis 类型：

```text
Round0Genesis:
  来自 encoder code 覆盖不足。

CompositionGenesis:
  来自 round r>0 的 FeatureEvent 新组合覆盖不足。

WeakParentRefinement:
  存在 weak parent，但 parent 不足以 fork 或覆盖当前上下文。

ClaimMismatchGenesis:
  多个高置信 claim 在同类证据上失败，提示存在未建模上下文。
```

这使“已知基础特征的新颖组合”可以触发 genesis，而不是被误判为已知 token/code 的普通变体。

### 9.4 Fork 动机

Fork 解决的是稳定结构遇到系统性冲突的问题。

```text
stable parent + conflicting context -> child branch
```

### 9.5 Fork 条件

Fork 只在 `parent_status = StableConflictParent` 时发生。

```text
parent.utility >= fork_utility_min
parent.resistance >= fork_resistance_min
parent.interference rising
current_context separable
overwrite_cost > fork_cost
child_expected_gain > child_rent
```

如果 parent 只是 WeakParent，则不 fork，而是进入 WeakParentRefinement genesis。这样 fork 不承担冷启动和弱结构补洞职责。

### 9.6 Child 初始化

```text
child.parent = parent.id
child.plasticity = higher than parent
child.resistance = lower than parent
child.output_head = parent output + local residual seed
child.segments = parent relevant segments + conflict-context sketch
child.phase = Probe
```

### 9.7 Parent 保护

Fork 后 parent 不被冻结为神圣结构，但其 plasticity 不应因当前冲突被强行提高。

```text
parent retains old contexts
child competes in new context
router must expose both fairly
```

### 9.8 Merge / Retire

Child 的结局：

```text
if child.utility positive and distinct:
    promote to Active

if child.utility positive but redundant with parent:
    merge useful segments

if child.utility negative:
    retire

if child causes interference:
    quarantine or split again only under strict budget
```

### 9.9 Fork/Genesis 指标

```text
genesis_attempts
genesis_survival_rate
genesis_rent_loss
fork_attempts
fork_survival_rate
fork_parent_damage
fork_child_gain
category_proliferation_rate
active_probe_count
retired_probe_count
```

---

## 10. Routing 与 Activation

### 10.1 Router 的设计立场

Router 是必要 scheduler，不是涌现智能。

第一版不使用复杂加权 scorer：

```text
score = w1*input_match + w2*segment_match + ...
```

这种形式会把手写先验藏在 routing 层，造成 router masking substrate failure。

### 10.2 Candidate sources

候选来源固定分 quota：

```text
encoder_bucket_quota
segment_match_quota
feature_event_quota
link_propagation_quota
recent_active_quota
probe/genesis_quota
exploration_quota
```

每个 source 都有最小曝光预算，防止某一路由源永久垄断。

### 10.3 Lexicographic scheduler

Source 内部排序只使用少量单调规则：

```text
1. phase allowed? Retired no, Quarantined limited
2. match above threshold?
3. positive rent-adjusted utility?
4. false-positive cost acceptable?
5. execution cost acceptable?
6. deterministic tie-break by stable id
```

不进行复杂线性组合。

### 10.4 Activation

激活使用固定 TopK：

```text
active_elements = SelectK(candidates, K)
```

但 SelectK 必须保留 source diversity：

```text
no source may occupy all active slots unless other sources empty
```

### 10.5 Router 审计指标

```text
candidate_source_exposure
activation_share_by_source
positive_utility_orphans
never_activated_surviving_structures
source_ablation_delta
router_counterfactual_recall
exploration_hit_rate
probe_starvation_rate
```

如果某类结构从未获得路由机会，则不能判定该结构机制无效。

---

## 11. Segment 与组合深度

### 11.1 Segment 作为 latent context 机制

Segment 的目标不是复现 token recurrence，而是检测 latent sparse state：

```text
previous sparse code
feature events
active elements
claim verification state
link context
```

### 11.2 第一版 Segment 匹配

```text
match_score = sum(active_synapse_weights)
matched = match_score >= threshold
```

要求：

```text
synapse count bounded
segment check bounded
no global scan
no target leakage
```

### 11.3 Segment 学习

```text
if segment matched and owner contributed positively:
    strengthen active synapses
    weaken false-positive synapses

if segment failed to match but owner should have been active:
    add sparse synapses from causal ledger / current code

if segment causes false positives:
    increase threshold or reduce utility
```

### 11.4 FeatureEvent emission

Segment match 可以产生 FeatureEvent：

```text
if segment.matched and segment.emits_feature_event:
    emit FeatureEvent(segment.feature_id)
```

FeatureEvent 进入下一 composition round。

### 11.5 Bounded composition rounds

```text
Round 0: Encoder sparse code
Round 1: Segment match over encoder/recent active
Round 2: Segment match over FeatureEvents + active elements
Round D: final active set for prediction
```

第一版配置：

```text
D in {1, 2, 3}
```

D 不是要调的性能超参，而是表达力 gate。

### 11.6 Composition 指标

```text
D1_vs_D2_delta
D2_vs_D3_delta
feature_false_positive_rate
feature_reuse_score
feature_rent_adjusted_utility
nested_task_success
xor_like_task_success
```

如果 D=2/3 不能在组合任务上超过 D=1，ESM 的深度表达力不足。

---

## 12. Local Predictor

第一版仍然使用 token cross-entropy 作为主要可比指标，但必须拆开输出贡献。

```cpp
struct OutputHead {
    OutputHeadKind kind;
    SparseLogitTable residuals;
    float mixture_weight;
    float utility;
    float false_positive_cost;
};
```

输出组成：

```text
global prior
current-token/hash control
encoder-only head
element residual head
feature-event head
claim head
```

必须报告：

```text
full ESM
no element residual
no feature event
no claim
encoder-only
hash-control-only
prior-only
```

防止 output readout masking substrate failure。

---

## 13. CPU 友好实现规划

### 13.1 Store 设计

逻辑 ID 与物理槽分离：

```cpp
ElementId -> physical_slot
physical_slot -> SoA arrays
```

核心数组：

```cpp
struct ElementStore {
    std::vector<float> plasticity;
    std::vector<float> resistance;
    std::vector<float> utility;
    std::vector<float> reliability;
    std::vector<float> interference;
    std::vector<uint32_t> segment_begin;
    std::vector<uint32_t> segment_count;
    std::vector<uint32_t> link_begin;
    std::vector<uint32_t> link_count;
    std::vector<uint8_t> phase;
};
```

### 13.2 Hot path 避免对象图

热路径禁止：

```text
per-step heap allocation
std::unordered_map lookup in inner loop
virtual dispatch in inner loop
full element scan
recursive graph traversal
```

热路径应使用：

```text
fixed-capacity candidate arrays
contiguous segment blocks
small sorted vectors
stable integer IDs
partial sort / heap select
bitset or sparse-set intersection
```

### 13.3 Index 设计

需要索引：

```text
encoder feature -> candidate elements
feature event -> segment list
recent active -> outgoing links
phase -> eligible pool
probe pool -> exploration candidates
```

索引必须增量维护，不得每步重建。

### 13.4 Composition cost

每步组合开销：

```text
composition_rounds <= D
features_per_round <= F
segments_checked_per_round <= S
```

Claim 也必须进入每步预算线。但预算不得写成与语料速率绑定的“每 N token”全局时钟；第一版使用相对结构预算：

```text
claims_issued_per_step <= floor(claim_issue_fraction * active_element_budget)
probe_claims_per_step <= floor(probe_issue_fraction * exploration_budget)
claims_checked_per_step <= floor(claim_check_fraction * active_element_budget)
claim_index_hits_per_step <= floor(claim_hit_fraction * segment_check_budget)
open_claims_total <= floor(open_claim_fraction * live_element_count)
```

这些 fraction 是硬件/结构预算比例，不是记忆时间尺度。它们控制每步 CPU 和内存占用，而不表达“多久以后忘记”。

记录：

```text
active_ops
segment_checks
feature_events_emitted
ledger_entries
claims_issued
claims_checked
open_claims
claim_rent_paid
cache_miss_proxy
bytes_touched
```

### 13.5 内存与诊断

每个结构都必须有 byte cost：

```text
element_bytes
segment_bytes
synapse_bytes
link_bytes
feature_event_bytes
claim_bytes
ledger_bytes
index_bytes
```

预测增益必须与 bytes 对齐报告。

---

## 14. 最小原型范围

### 14.1 必须实现

```text
Sparse encoder A/B/C
ElementStore
SegmentStore
LinkStore
FeatureEvent buffer
PredictionLedger
PendingClaim pool
ClaimIssuer with hard budgets
Quota router
Lexicographic scheduler
Local predictor
Plasticity/resistance update
Genesis lifecycle
Fork lifecycle
Composition rounds D=1/2/3
Strict prequential runner
Metrics logger
Checkpoint/resume
```

### 14.2 明确不实现

第一版不实现：

```text
multi-brain-region modules
external replay schedule
dense embeddings
global backprop
ANN retrieval
Transformer attention
GPU dependency
large corpus training
RL environment
complex natural language generation
```

### 14.3 与旧 SBM 的关系

旧 SBM 可作为工程资产与 baseline：

```text
mmap corpus runner
checkpoint/resume
frozen evaluation discipline
C/Python ABI 思路
structural lifecycle 审计精神
current-token / n-gram / residual baselines
```

但 ESM 不应继续继承地址程序作为主干。

---

## 15. 实验路线

## Gate E-1A: Sparse Representation Quality

目的：验证 encoder 不只是 token hash。

任务：

```text
same token, different latent roles
different tokens, same predictive role
context switch
noisy sequence class
```

对照：

```text
raw token/hash encoder
online sparse competitive encoder
predictive sparse encoder
```

通过条件：

```text
learned sparse encoder > raw hash encoder
控制 token id / ngram 后仍有 predictive gain
same-token context split 可观测
role sharing 可观测
usage entropy 不坍缩
```

## Gate E-1B: Causal Eligibility / Delayed Evidence

目的：验证系统能处理延迟信用。

任务：

```text
A ... ... ... B -> target Y
latent cue at t-k, verification at t
```

对照：

```text
instant-only local update
causal ledger update
pending-claim update
```

通过条件：

```text
instant-only fails
ledger / claim succeeds
path_responsibility 可由 ledger DAG 直接计算
causal_confidence 使用结构类型的具体公式而非占位符
claims_issued_per_step / claims_checked_per_step 有硬预算
no target leakage
ledger bytes bounded
claim rent prevents unlimited memory
```

## Gate E-1C: Composition Depth

目的：验证系统不是单层阈值检测器。

任务：

```text
A and B imply C
C and D imply target
XOR-like latent composition
nested dependency
```

对照：

```text
D=1
D=2
D=3
```

通过条件：

```text
D=2 or D=3 > D=1
feature events have positive rent-adjusted utility
feature false-positive rate controlled
active compute bounded
```

## Gate E-1D: Genesis / Cold Start

目的：验证无父结构时系统能启动。

任务：

```text
empty field stream
novel pattern stream
rare event stream
```

通过条件：

```text
genesis creates probe structures
probe survival rate nonzero but bounded
probe explosion controlled by rent
no-parent case not stuck
weak-parent gap does not block learning
round>0 composition gaps can trigger genesis
```

## Gate E-1E: Router Fairness

目的：验证 router 没有掩盖 substrate。

通过条件：

```text
all candidate sources receive exposure
positive-utility structures are not starved
source ablation report available
router counterfactual recall above floor
```

## Gate E0: 机制正确性

目标：证明实现没有 leakage，所有成本可计量。

通过条件：

```text
prediction-before-target invariant
active budget invariant
candidate budget invariant
segment check budget invariant
composition budget invariant
ledger budget invariant
checkpoint determinism
```

## Gate E1: Elastic plasticity 优于 fixed plasticity

对照：

```text
fixed plasticity
age-based plasticity
utility/interference-based plasticity
```

通过条件：

```text
elastic plasticity improves post-shift recovery
old-rule retention not worse
resistance correlates with held-out utility
```

## Gate E2: Fork-before-overwrite

任务：

```text
stable rule A
shift to conflicting rule B
later return to A
```

通过条件：

```text
fork model retains A better than overwrite model
fork child learns B faster than no-fork model
fork count bounded
category proliferation controlled
```

## Gate E3: Latent context beats token recurrence

任务：

```text
token recurrence misleading
latent role predictive
same surface token different role
```

通过条件：

```text
ESM latent segment > token recurrence baseline
segment utility not explained by token id alone
encoder ablation damages result
```

## Gate E4: 小型真实语言流

不直接上大语料。先用小型真实流：

```text
technical prose
dialogue fragments
code comments
short wiki passages
```

指标：

```text
prequential NLL
post-topic-shift recovery
rare-token recovery
entity recurrence
bytes_per_predictive_gain
active_ops_per_predictive_gain
```

---

## 16. 指标体系

### 16.1 Predictive metrics

```text
NLL
bits/token
ΔNLL vs prior
ΔNLL vs current-token control
ΔNLL vs ngram/hash control
calibration error
```

### 16.2 Representation metrics

```text
code entropy
usage Gini
same-token split
role-sharing score
predictive mutual information after token control
encoder ablation delta
```

### 16.3 Eligibility metrics

```text
ledger_credit_mass
delayed_credit_success
pending_claim_verified_rate
pending_claim_false_alarm_rate
claim_rent_loss
claim_survival_rate
```

### 16.4 Elasticity metrics

```text
plasticity distribution
resistance distribution
resistance-utility correlation
plasticity-churn rate
high-utility overwrite rate
```

### 16.5 Structural metrics

```text
element count
segment count
link count
feature event count
claim count
genesis rate
fork rate
merge rate
retire rate
quarantine rate
```

### 16.6 Router metrics

```text
candidate source exposure
activation share by source
positive-utility orphan rate
probe starvation rate
source ablation delta
router counterfactual recall
```

### 16.7 CPU metrics

```text
active elements / step
candidates / step
segment checks / step
composition rounds / step
ledger entries / step
bytes touched / step
wall-clock tokens/sec
RSS memory
stored bytes per ΔNLL
active ops per ΔNLL
```

---

## 17. 失败模式

### 17.1 Encoder collapse

症状：少数 encoder columns 垄断全部输入。

诊断：

```text
usage entropy low
same-token split absent
role sharing absent
learned encoder ≈ hash control
```

处理：

```text
homeostatic correction
anti-correlation pressure
retire useless columns
increase genesis only if rent-positive
```

### 17.2 Token-hash disguise

症状：系统看似有 latent segment，但其预测增益完全由 token id / ngram 解释。

诊断：

```text
controlled predictive MI = 0
encoder ablation no effect
hash control matches full model
```

处理：停止进入后续 Gate，先修 encoder。

### 17.3 Plasticity collapse

症状：所有结构 resistance 过高，不再适应。

处理：

```text
negative utility lowers resistance
interference prevents blind freezing
rent prevents useless frozen structures
```

### 17.4 Plasticity churn

症状：结构一直改写，没有稳定知识。

处理：

```text
positive utility increases resistance
held-out utility protects stable structures
fork-before-overwrite isolates conflict
```

### 17.5 Genesis explosion

症状：每个新输入都创建 probe element。

处理：

```text
strict genesis quota
probe rent
coverage threshold
survival requirement
retire low-utility probes
```

### 17.6 Fork explosion / category proliferation

症状：每个冲突都生成 child，结构数量爆炸。

处理：

```text
fork cost
child rent
merge redundant child
minimum separability requirement
parent damage accounting
```

### 17.7 Silent overwrite

症状：高价值结构被新样本慢慢污染。

处理：

```text
overwrite cost accounting
old-context probes
high-resistance protection
fork pressure increase under systematic conflict
```

### 17.8 Segment false-positive dominance

症状：某些 segment 到处 firing，提升短期 recall 但损害长期泛化。

处理：

```text
false-positive cost
precision EMA
threshold adjustment
quarantine segment
```

### 17.9 Router masking substrate failure

症状：某结构机制看似无效，但其实 router 从未给它机会。

诊断：

```text
source exposure low
positive-utility orphan high
probe starvation high
source ablation inconclusive
```

处理：

```text
quota scheduler
source fairness report
router counterfactual recall
```

### 17.10 Composition absence

症状：D=2/3 不优于 D=1，feature events 无正 utility。

解释：系统本质上仍是单层阈值检测器。

处理：重新设计 FeatureEvent 与 Segment input，而不是扩大模型。

### 17.11 Output readout masking substrate failure

症状：NLL 改善来自 global prior / residual head，而不是 substrate。

诊断：

```text
encoder-only / element-only / feature-only / claim-only ablations
```

处理：拆分报告，不允许 full metric 单独作为成功证据。

---

## 18. 代码组织建议

```text
esm/
  include/esm/
    types.hpp
    config.hpp
    element_store.hpp
    segment_store.hpp
    link_store.hpp
    encoder.hpp
    feature_event.hpp
    prediction_ledger.hpp
    pending_claim.hpp
    router.hpp
    predictor.hpp
    plasticity.hpp
    lifecycle.hpp
    metrics.hpp

  src/
    encoder/
      raw_hash_encoder.cpp
      sparse_competitive_encoder.cpp
      predictive_sparse_encoder.cpp
    store/
      element_store.cpp
      segment_store.cpp
      link_store.cpp
    router/
      quota_router.cpp
      lexicographic_scheduler.cpp
    composition/
      feature_event_buffer.cpp
      composition_rounds.cpp
    learning/
      causal_ledger.cpp
      pending_claim.cpp
      plasticity_update.cpp
      genesis.cpp
      fork.cpp
      retire.cpp
    predictor/
      local_predictor.cpp
      output_heads.cpp
    runner/
      prequential_runner.cpp
      synthetic_tasks.cpp
      language_stream_runner.cpp
    metrics/
      metrics_logger.cpp
      audit_report.cpp

  experiments/
    gate_e_minus_1a_encoder.json
    gate_e_minus_1b_eligibility.json
    gate_e_minus_1c_composition.json
    gate_e_minus_1d_genesis.json
    gate_e_minus_1e_router.json
    gate_e0_correctness.json
    gate_e1_elasticity.json
    gate_e2_fork.json
    gate_e3_latent_context.json
    gate_e4_language_stream.json
```

---

## 19. 默认配置哲学

### 19.1 不调参优先

第一版默认配置应尽可能离散化：

```text
plasticity bucket: Low / Medium / High
resistance bucket: Low / Medium / High
utility bucket: Negative / Neutral / Positive
interference bucket: Low / High
```

更新使用符号规则，而不是连续超参精调。

### 19.2 Quota 优先于加权 scorer

```text
bad:
  hand-tuned linear score over many features

better:
  fixed source quotas + simple monotonic ordering
```

### 19.3 Gate 优先于性能

第一阶段不追求低 NLL，而追求机制可证伪：

```text
encoder 是否真有 latent value
eligibility 是否真能跨步分配 credit
genesis 是否真能冷启动
router 是否公平
composition 是否真有深度
```

### 19.4 CPU 预算不得事后补救

任何机制加入时必须同时写出：

```text
per-step active cost
storage cost
index cost
diagnostic metric
ablation plan
failure mode
```

---

## 20. 与 Transformer 对比的正确方式

ESM 不应在第一阶段说：

```text
我要用 CPU 稀疏结构直接击败 Transformer perplexity。
```

正确对比制度是：

```text
online stream
prediction-before-target
fixed active compute
growing persistent memory
no dense backprop
no epoch retraining
post-shift recovery
storage efficiency
```

对照模型：

```text
unigram / global prior
current-token control
small ngram
hash residual table
old SBM strongest control
small RNN / small Transformer if feasible
```

ESM 的初期成功标准：

```text
在 fixed active compute 下，随着 stored structure 增长，
ESM 获得的 rent-adjusted predictive gain 超过 hash/ngram/SBM controls，
且增益来自 encoder/segment/composition/fork 等 substrate，而不是 output readout。
```

---

## 21. 当前规划结论

ESM Draft 0.2 的核心结论：

1. ESM 不是空白机制空间，必须明确承认 HTM / ART / continual learning / threshold circuit 的前史。
2. ESM 的新意是 CPU-first、严格因果、成本审计、可归因 gate、genesis/fork 分离、router 审计、composition-capable sparse substrate。
3. `encode / signature formation` 是最大前置风险，必须作为 Gate E-1A，而不是留给后续自然涌现。
4. 瞬时局部更新不够，必须通过 causal ledger 与 pending claim 处理延迟信用。
5. Fork 不等于 genesis。系统必须能在无 parent 时冷启动新结构。
6. Router 是技术债，第一版必须使用可审计 quota scheduler，而不是复杂手写权重 scorer。
7. Segment 必须能产生可引用 FeatureEvent，否则系统表达力停留在平坦阈值检测器。
8. “非幂律”必须替换为 predictive gain per active op / per stored byte 等可证伪指标。
9. Gate E-1 不通过，不进入真实语言流。

---

## 22. 下一步

下一步不是直接写完整模型，而是实现 Gate E-1 的最小 synthetic runner。

优先顺序：

```text
1. Gate E-1A: encoder quality
2. Gate E-1B: causal ledger / pending claim
3. Gate E-1C: composition depth
4. Gate E-1D: genesis cold start
5. Gate E-1E: router fairness
```

只有这五个前置问题闭合，ESM 才值得进入原始 E0/E1/E2/E3/E4。

---

## 23. Draft 0.3 新增的可实现规则摘要

本节把 Draft 0.3 的关键补丁压缩成实现 checklist。

### 23.1 Encoder B/C 是独立研究风险

```text
E-1A 不通过:
  不进入语言流。

E-1A 失败但其他 gate 通过:
  结论写成 substrate 可运行，但在线 sparse representation formation 未闭合。
```

### 23.2 Claim 发放预算

```text
Claim 只能由 ClaimIssuer 创建。
ClaimIssuer 只能在 Adapt phase 运行。
Claim 不得影响当前预测。
Claim 必须受 C_issue / C_probe / C_check / C_hit / C_open 预算限制。
```

### 23.3 Genesis/Fork 联合边界

```text
先计算 parent_status。
NoAdequateParent -> genesis。
WeakParent -> weak-parent rescue genesis + parent adaptation。
StableConflictParent -> fork。
StableCompatibleParent -> ordinary adaptation。
```

WeakParent 不得阻止 genesis。

### 23.4 Genesis 来源

```text
Genesis source pattern may come from any round r in 0..D.
Round 0 handles encoder-code novelty.
Round r>0 handles novel composition of known features.
```

### 23.5 Ledger credit 最小算法

```text
path_responsibility:
  backward mass propagation on bounded ledger DAG.

causal_confidence:
  ordinal bucket rule from precision / resolution / stability; no hidden weighted scorer.

credit normalization:
  total magnitude capped by error_budget_t.
```

---

## 24. Draft 0.4 原则一致性修正

本节覆盖研究报告中出现但不得直接进入代码的三类占位符。它们不是 ESM 机制的一部分，除非未来通过独立 gate 证明。

### 24.1 禁止隐藏线性加权 confidence

禁止以下形式进入默认实现：

```text
conf = a * metric_1 + b * metric_2 + c * metric_3
```

原因：这与 router 层禁止多项手写 scorer 的原则相同。隐藏线性权重会把设计者偏好伪装成结构涌现。默认实现必须使用离散序数规则：

```text
metric buckets = {Low, Unknown, High}
if any critical metric is Low: confidence = Low
elif two or more independent metrics are High: confidence = High
else: confidence = Medium
```

数值化只发生在最后 credit magnitude 阶段，不参与排序、路由或结构创建。

### 24.2 Rescue 不得使用固定 occurrence/token 窗口

禁止以下形式进入默认实现：

```text
weak_count(parent) >= N within last 64 occurrences
genesis_per_1k_tokens_max = K
forks_per_parent_per_1k_tokens_max = K
```

这些都是外部时钟。替代规则使用结构相对量：

```text
weak_pressure(parent) = weak_mismatch_mass(parent) / max(parent_exposure_mass, eps)
conflict_pressure(parent) = confirmed_conflict_mass(parent) / max(parent_success_mass + confirmed_conflict_mass, eps)
genesis_pressure(pattern) = unexplained_mass(pattern) / max(exposure_mass(pattern), eps)
```

Rescue 决策：

```text
if parent_status == WeakParent
and weak_pressure(parent) is High
and current pattern has Low coverage
and rescue_budget_share available:
    allow RescueGenesis or RescueFork
```

结构增长预算也使用相对 live state：

```text
max_probe_elements = floor(probe_fraction * live_element_count) + bootstrap_probe_floor
max_children_per_parent = function(parent_reliable_utility_bucket, parent_interference_bucket)
growth_budget_this_step = floor(growth_fraction * active_element_budget)
```

这里 `bootstrap_probe_floor` 是冷启动所需的结构下限，不是时间尺度。

### 24.3 Genesis / Fork / Rescue 联合边界的最终规则

每步先计算：

```text
coverage_bucket(pattern) in {Low, Medium, High}
novelty_bucket(pattern) in {Low, Medium, High}
weak_pressure_bucket(parent) in {Low, Medium, High}
conflict_pressure_bucket(parent) in {Low, Medium, High}
parent_status in {NoAdequateParent, WeakParent, StableConflictParent, StableCompatibleParent}
```

决策表：

```text
NoAdequateParent + novelty High + coverage Low:
  RoundGenesis or CompositionGenesis

WeakParent + weak_pressure High + coverage Low:
  WeakParentRefinementGenesis

WeakParent + conflict_pressure High + parent has reusable partial structure:
  RescueFork

StableConflictParent + conflict_pressure High + overwrite_cost > fork_cost:
  Fork

StableCompatibleParent:
  ordinary adaptation only
```

WeakParent 不能 veto genesis。StableCompatibleParent 才能 veto genesis。

### 24.4 Claim 发放主体与预算的最终规则

ClaimIssuer 只允许在 Adapt phase 运行，且不得影响当前 prediction。每步 claim budget 来自 active budget：

```text
C_issue = floor(issue_fraction * K_active)
C_probe = floor(probe_fraction * K_explore)
C_check = floor(check_fraction * K_active)
C_open  = floor(open_fraction * live_element_count) + bootstrap_claim_floor
```

Claim 发放条件：

```text
TemplateClaim:
  matched ClaimSegment
  and issuer confidence not Low
  and expected_future_evidence exists
  and rent_adjusted_expected_value bucket not Negative

ProbeClaim:
  coverage Low
  and novelty or composition_gap High
  and probe budget available
  and no equivalent open claim exists

RescueClaim:
  WeakParent or ClaimMismatchGenesis candidate
  and evidence template can distinguish parent vs candidate
```

Claim 去重键：

```text
(issuer_class, expected_future_evidence_key, source_round, parent_status)
```

等价 claim 已开放时，只增加支持质量，不新建 claim。

### 24.5 E-1A 不是固定工期阶段

E-1A 分为两个阶段：

```text
E-1A-build:
  实现 synthetic generators、hash control、Encoder B、Encoder C、诊断日志。
  这是工程阶段，可以有短期实现目标。

E-1A-research:
  只以 gate 通过/失败/重设为退出条件。
  不以固定周数推进到 E-1B。
```

Stop rule：

```text
if Encoder B/C do not beat raw hash control on same-token-context-split
and do not show controlled predictive mutual information above token/ngram controls:
    stop architecture scaling
    redesign encoder
```

因此路线图不再把 Phase A 与 ledger/router/genesis 工程阶段并列估算。E-1A 是科学风险；E-1B/E-1D/E-1E 才主要是工程正确性问题。

### 24.6 引用清理规则

任何会进入架构决策的外部数字必须在文档中写成作者/年份/论文名或链接，不允许保留工具内部引用残留。高精度数字只允许用于“佐证方向”，不得成为 ESM 默认参数。

已复核的数字示例：

```text
Khandelwal et al. 2020, kNN-LM:
  WikiText-103 perplexity 15.79, 2.9 point improvement, no additional training.

Iyer et al. 2022, Active Dendrites:
  Meta-World MT10 Experiment 1 overall success 87.5% vs MLP baseline 76.6%.

OpenSearch 2024 neural sparse search blog:
  Lucene upgrade reduced P99 latency by 72% / 76% for doc-only / bi-encoder modes in their reported setup.
```

这些数字不得被转译成 ESM 规格参数。

---

## 25. 参考坐标

本规划书不是文献综述，但以下研究坐标必须作为后续设计审计背景：

1. Subutai Ahmad, Jeff Hawkins, “Properties of Sparse Distributed Representations and their Application to Hierarchical Temporal Memory”, 2015.
2. Jeff Hawkins, Subutai Ahmad, “Why Neurons Have Thousands of Synapses, A Theory of Sequence Memory in Neocortex”, 2015.
3. Yuwei Cui, Subutai Ahmad, Jeff Hawkins, “Continuous Online Sequence Learning with an Unsupervised Neural Network Model”, Neural Computation, 2016.
4. Stephen Grossberg / Gail Carpenter, Adaptive Resonance Theory and fuzzy ART / ARTMAP literature.
5. Temporal-difference learning and eligibility trace literature.
6. Threshold circuit / linear separability literature.
7. Kaplan et al., “Scaling Laws for Neural Language Models”, 2020.
8. Hoffmann et al., “Training Compute-Optimal Large Language Models”, 2022.
9. Sparse semantic representation / Semantic Folding / sparse word vector literature.
10. Competitive Hebbian / anti-Hebbian sparse representation learning and homeostatic sparse coding literature.
11. Growing Neural Gas / growing-when-required style incremental structure creation literature.

