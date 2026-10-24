# 美妆客服话术 SFT 微调 Demo

面试演示项目：美妆电商客服话术 SFT 全链路 demo，覆盖四大模块——数据管线（模拟千牛数据生成 → 脱敏 → 会话拆分 → 优质筛选 → 分类配比 → 占位符化 → 训练集构建）、LoRA 训练（Qwen2.5-7B-Instruct + peft/trl，AutoDL 4090）、在线辅助应答系统（意图分类两版 / 事实注入 / 数字校验 / 双条件闸门 / 大促规则 / Streamlit 演示界面）、三层评估（BLEU / 合规率 / 盲测问卷）。

## 目录结构

```
src/
  config.py            # 集中配置（环境变量、数据路径、业务常量）
  data_pipeline/       # 数据管线：generate_raw / desensitize / split_sessions /
                       #   filter_quality / label_category / build_dataset / run_pipeline
  training/            # LoRA 训练（train_lora.py）与推理服务（infer_server.py）
  serving/             # 在线辅助应答：intent / fact_apis / generator / pipeline /
                       #   promo_knowledge / stats / web_app（Streamlit 界面）
  evaluation/          # 三层评估：bleu_eval / compliance / blind_test
tests/                 # pytest 用例（T 编号对应 spec/test_cases.md）
spec/                  # 行为规格与测试用例（SpecDD）
scripts/               # 早期清洗管线（配套 面试问答-数据清洗流程.md）
data/
  raw/                 # 模拟千牛原始数据
  processed/           # 管线各阶段产物与报告
  train/               # 训练集 train.jsonl / 测试集 test.jsonl
  clean/               # 早期清洗管线产出（面试问答素材）
requirements.txt       # 本地 demo 环境依赖
requirements-train.txt # 训练环境依赖（AutoDL 4090）
```

## 环境准备

- Python 3.10+
- 安装本地 demo 依赖：`pip install -r requirements.txt`
- 环境变量：
  - `AGICTO_API_KEY`（必需）：AGICTO 平台（https://agicto.com，OpenAI 兼容接口）key，离线 LLM 打标（T6/T7）与在线意图分类 V2 使用，未设置时相关功能直接报错，不做静默降级
  - `LLM_BASE_URL`（可选）：接口地址，默认 `https://api.agicto.cn/v1`
  - `LLM_MODEL`（可选）：默认 `deepseek-flash`（已验证适合打标/分类等轻量高频任务）
  - `GENERATE_API_URL`（可选）：AutoDL 远程推理服务地址，默认 `http://127.0.0.1:8001`

## 演示步骤（面试演示剧本）

### 第一步：数据管线

分步执行（每步对应一节叙事，产物落盘便于逐环展示）：

```
python -m src.data_pipeline.generate_raw      # T3 生成模拟千牛原始数据
python -m src.data_pipeline.desensitize       # T4 脱敏（手机号/地址/订单号/昵称 → 占位符）
python -m src.data_pipeline.split_sessions    # T5 会话拆分（同客户间隔 30 分钟切分）
python -m src.data_pipeline.filter_quality    # T6 优质筛选（基础门槛 + 快速通道 + LLM 多信号）
python -m src.data_pipeline.label_category    # T7 LLM 打标与 60/30/10 配比
python -m src.data_pipeline.build_dataset     # T8 价格占位符化 + 训练集/测试集分层构建
```

全流程一键：`python -m src.data_pipeline.run_pipeline`

注意：T6/T7 两环调用真实 LLM，需先设置 `AGICTO_API_KEY`。产物依次落在 `data/raw/`、`data/processed/`（含各级漏斗报告）、`data/train/`。

### 第二步：启动演示界面

```
streamlit run src/serving/web_app.py
```

浏览器访问 `http://localhost:8501`。侧边栏预设五个演示场景（一键填入输入框），按顺序讲：

1. **价格咨询自动发送**："这款多少钱？有优惠券吗"——意图判为价格咨询且高置信 → 商品接口注入现价/券等事实 → 生成回复中的数字全部来自接口 → 数字校验通过 → 双条件闸门**自动发送**。
2. **意图版本对比（踩坑故事）**："现在买划算吗"——没有任何价格关键词，V1 规则漏判为"其他"，V2 LLM 正确判为"价格咨询"并给出置信度；侧边栏切换 V1/V2 两版对比，讲为什么从规则迭代到 LLM（V2 需 `AGICTO_API_KEY`）。
3. **高危永远转人工**："我用了你们的面霜过敏了！烂脸了！"——命中过敏/烂脸类高危意图，不进入生成，直接转人工，强调安全底线优先于自动化率。
4. **物流查询**："我的快递到哪了"——快递接口按订单号注入物流状态/最新节点/预计送达，同样受事实校验约束。
5. **大促规则（需先上传规则）**：在看板上传双 11 规则文档（界面提供示例一键填入）→ 条款切块并标注有效期 → 提问"双11定金和尾款怎么算，能叠加吗"命中条款注入生成上下文 → 若模型编造规则外优惠（如"满 200 减 30"）被数字/时间校验拦截 → 有效期过后规则自动下线，检索为空，链路转人工。

说明：本地演示生成器选"本地模拟"（MockGenerator，忠实引用注入事实）；真实 SFT 模型效果需完成下文 AutoDL 部署后，在侧边栏切换"远程推理服务"。

### 第三步：三层评估

候选生成结果文件约定：与测试集逐行一一对应，每行一条 `{"reply": ...}`（先用推理服务把测试集首问跑成结果文件）。

```
# 合规率（默认读测试集，批跑禁用语/称呼规范/emoji 上限，报告落 data/processed/）
python -m src.evaluation.compliance --input data/train/test.jsonl [--base-input <通用模型回复.jsonl>]

# BLEU 趋势对比（SFT 前后两组候选，--base-candidates 可选）
python -m src.evaluation.bleu_eval --candidates <SFT生成结果.jsonl> [--base-candidates <通用模型结果.jsonl>]

# 盲测问卷（抽 100 题，SFT/通用/真人三答案打乱，问卷与答案映射落 data/processed/）
python -m src.evaluation.blind_test --sft <SFT结果.jsonl> --generic <通用结果.jsonl>
```

## AutoDL 训练与部署（用户人工执行）

1. 开通 AutoDL 单卡 RTX 4090（24GB）实例，镜像选 PyTorch 2.x（CUDA 12.x）。
2. 上传项目代码与 `data/train/train.jsonl`，安装训练依赖：`pip install -r requirements-train.txt`。
3. 模型下载（二选一）：实例可访问 HuggingFace 则直接用默认 `Qwen/Qwen2.5-7B-Instruct`；无法直连则开启学术资源加速，或用 modelscope 预下载后以 `--model_name <本地路径>` 启动（详见 `src/training/train_lora.py` 头部注释）。
4. 训练：`python src/training/train_lora.py --data_path data/train/train.jsonl`（默认 3 轮、batch size 4，adapter 输出 `autodl-tmp/adapter`）。
5. 启动推理服务：`python src/training/infer_server.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_path autodl-tmp/adapter --port 8001`。
6. 本地演示界面侧边栏将生成器模式切换为"远程推理服务"，并用环境变量 `GENERATE_API_URL` 指向实例服务地址（默认 `http://127.0.0.1:8001`，远程实例需自行配置端口映射/隧道）。

预期规模：训练集约 800 条，4090 单卡小时级跑完，实例费用十元级。

## 测试

```
python -m pytest tests/ -q
```

当前 112 条用例全绿（用例与执行记录见 `spec/test_cases.md`）。
