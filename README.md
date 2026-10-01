# mini-harness (M4)

最小可用的 Agent 运行时：统一消息模型 · 双协议 Provider（Anthropic / OpenAI 兼容）· 退避重试 · 工具系统（校验/权限/超时/并发/幂等重试）· 带护栏的 Agent Loop · 运行监督 · **上下文管理（自动压缩 + prompt caching 断点）**。

## 快速开始（uv）
```bash
uv sync                                    # 按 uv.lock 创建 .venv（含 dev 组）
uv run pytest -q                           # 68 个测试，无需 API Key
uv run ruff check . && uv run ruff format --check .
uv run python examples/cli.py --fake       # 离线演示
ANTHROPIC_API_KEY=... uv run python examples/cli.py
OPENAI_API_KEY=... uv run python examples/cli.py --provider openai --model gpt-4o
```

## 用法
```python
from mini_harness import Agent, ContextConfig
agent = Agent(
    RetryingProvider(AnthropicProvider(key)), registry, model="...",
    context=ContextConfig(context_window=200_000),   # 请按所用模型显式设置
)
async for ev in agent.run("...", session):
    if isinstance(ev, Compacted): print(ev.stage, ev.tokens_before, "->", ev.tokens_after)
```
关闭压缩：`ContextConfig(enabled=False)`。摘要可换更便宜的模型/实现：`Agent(..., compactor=LLMSummarizer(provider, "small-model"))`，或自定义 `Compactor`。

## M4 新增
| 能力 | 位置 | 行为 |
|---|---|---|
| 大小判定 | `core/context.py` | 以**提供方上次返回的真实 prompt 用量**为基准，加上此后新增消息的估算；无用量时退化为纯估算（`core/tokens.py`，CJK 按约 1 token/字） |
| 触发与目标 | `ContextConfig` | 超过窗口 80% 触发，压到 50% 以下；最新 25% 窗口内的消息受保护，不动 |
| 切分 | `find_cut` | 切点绝不落在 `tool` 消息上（不拆散 tool_use/result 配对），且至少保留最近一个 assistant/user 单元；单个超长 run（只有一条 user 消息）也能压缩 |
| 阶段 1：裁剪 | `trim_tool_results` | 旧的超长工具输出只留预览 + 标记（含工具名与裁掉的字符数）；确定性、幂等；够用就不调模型 |
| 阶段 2：摘要 | `LLMSummarizer` | 把旧段折叠成**一条**结构化摘要（目标/已完成/关键事实/精确标识符/待办）；已有摘要会被并入（滚动，不堆叠）；转录内容当作数据而非指令 |
| 降级 | `ExtractiveCompactor` | 摘要模型不可用或返回空时，用无模型的机械摘要，保证 run 不中断 |
| 原文不丢 | `Session.archive` | 被裁剪/折叠的原始消息进入 archive（M5 持久化），不再发送给模型 |
| 可观测 | `Compacted` 事件 | stage（trim/summary/extractive）、前后 token、归档条数；摘要花费计入 `session.usage` |
| Prompt caching | `CachePlan` + Anthropic 适配器 | 显式断点：静态前缀（tools+system）、稳定摘要、最新消息，≤4 个；OpenAI 适配器忽略（其前缀缓存自动生效） |
| 前缀稳定性 | 设计约束 + 测试 | 两次压缩之间历史严格追加；tools/system 不随轮变化；压缩只在越线时一次性发生，之后前缀重新稳定（有字节级测试） |

## 结构
```
mini_harness/
├── core/       messages · loop · context · compaction · tokens · limits · pricing · retry · guards
│               supervisor · session · events · errors
├── providers/  base · _sse · anthropic · openai_compat · retry · fake
├── tools/      spec · registry · executor · policy · builtin
└── sdk.py      Agent 门面（组装根）
```

## 已知局限（有意保留）
| 项 | 说明 | 后续 |
|---|---|---|
| 估算误差 | 无真实 tokenizer；靠"上次真实用量 + 增量估算"校准，但压缩后的"之后"大小仍是估算 | 可接入厂商 count_tokens 接口 |
| 压缩会使该次缓存失效 | 这是有意的代价：换来之后多轮的稳定前缀 | 阈值/目标可调 |
| 上下文超长被服务端拒绝 | 未做"强制压缩后重试一次" | 可作为 M4.1 |
| 摘要质量 | 取决于摘要模型；关键信息丢失需靠评测集发现 | M6 评测 |
| 受保护尾部本身过大 | 若最近消息已逼近阈值则无旧段可压，直接继续 | 配合工具结果截断（20k 字符上限） |
| 持久化 / 回退 | 内存 Session（archive 已就位） | M5 |
