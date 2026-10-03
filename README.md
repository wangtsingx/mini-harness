# mini-harness (v0.7 · M1–M6 完成)

一个最小可用、可复用的 Agent 运行时（Harness）。核心只有四件事：**模型 + 消息历史 + 工具 + 循环**；其余（重试、压缩、持久化、追踪、评测、MCP）都是可插拔扩展。

## 快速开始（uv）
```bash
uv sync                                    # 按 uv.lock 创建 .venv（含 dev 组）
uv run pytest -q                           # 195 个测试，无需 API Key
uv run ruff check . && uv run ruff format --check .

uv run python examples/cli.py --fake                       # 离线演示
uv run python examples/eval_demo.py                        # 评测 → 失败归因 → 基线对比
ANTHROPIC_API_KEY=... uv run python examples/cli.py        # 真实模型（REPL）
OPENAI_API_KEY=...    uv run python examples/cli.py --provider openai --model gpt-4o
```
CLI 开关：`--db/--session/--list-sessions`（持久化）· `--trace`（追踪树+指标）· `--record/--replay`（录制与回放）· `--mcp NAME=CMD` + `--grant exec`（接 MCP 服务器）。

## 能力地图
| 层 | 模块 | 要点 |
|---|---|---|
| 消息模型 | `core/messages` | 厂商无关；**tool_use/tool_result 配对不变量**，每次请求前校验，取消/崩溃后自动修复 |
| Provider | `providers/` | Anthropic、OpenAI 兼容、Fake；统一流事件；`RetryingProvider`（退避+抖动，流中断=丢弃重放） |
| 工具 | `tools/` | 装饰器注册 + pydantic Schema；权限策略；超时；并行（只读默认并行）；幂等工具仅在瞬时失败时重试 |
| Loop | `core/loop`, `supervisor` | 轮次/Token/成本/重复调用护栏；硬超时与取消；背压 |
| 上下文 | `core/context`, `compaction` | 以真实用量为基准的大小判断；先裁剪后摘要，摘要失败降级；Anthropic 缓存断点；前缀稳定 |
| 会话 | `session/` | SQLite 检查点（不可变快照+内容寻址）；回退=新分支；kill -9 后可恢复；乐观并发 |
| 可观测 | `observability/` | Run→Turn→{Model,Tool,Compaction} span；指标全部由 span 推导；脱敏 |
| 回放 | `eval/recording`, `replay` | 录下模型响应，用当前代码零成本重放，逐字段比对提示词与行为 |
| **评测** | `eval/runner`, `checks`, `attribution` | 并发运行、repeats 暴露 flaky、失败按层归因并聚类、基线对比 |
| **MCP** | `mcp/` | stdio JSON-RPC 客户端；服务器工具包装成普通 `Tool`；默认最小权限 |

## M6b：离线评测
```python
from mini_harness.eval import EvalCase, EvalRunner, contains, matches, all_of, tool_called

cases = [
    EvalCase("capital", "法国首都是？", all_of(tool_called("lookup"), contains("巴黎")), tags=("kb",)),
    EvalCase("format", "只回答数字：6*7", matches(r"^\d+$")),
]
runner = EvalRunner(lambda case, tracer: make_agent(tracer), cases, concurrency=4, repeats=3)
report = await runner.run()
print(report.render())  # 成功率、延迟 p50/p95、轮数、token、缓存命中、成本、按 tag、失败聚类
report.save("baseline.json")  # 之后的 PR：
cmp = new_report.compare("baseline.json", tolerance=0.34)  # 允许 3 次里偶发 1 次
assert cmp.ok, cmp.render()  # 回归的 case / 改进的 case / 成本与 p95 变化
```
**失败归因**（`eval/attribution.py`）是分诊而非证明：规则按优先级执行，第一条命中即返回，并附置信度与证据。

| 层 | 触发依据（示例） | 对应修法 |
|---|---|---|
| `infra` | `ProviderError`（鉴权、配额、宕机） | 修环境，不是修 Agent |
| `eval` | 检查函数自己抛异常 | 修测试 |
| `harness` | 非 Provider 异常；权限策略拒绝；护栏触发（max_turns/预算）且无工具报错 | 修运行时 / 调限额 / 调权限 |
| `context` | 检查函数声明的 `facts` **只存在于被压缩掉的历史中**（高）；或失败 run 里发生过压缩（低） | 调压缩参数 / 让摘要保留精确标识符 |
| `tool` | 工具报错（运行时错误/超时=高；参数非法=中）；护栏触发时工具一直失败 | 修工具：描述、Schema、错误信息 |
| `prompt` | 检查标注 `format`；或没调用任何工具就反问用户 | 澄清指令、加示例 |
| `model` | 剩余项：健康运行但答案错；重复同一调用卡死；调用不存在的工具 | 换更强模型 / 加示例 / 拆任务，先用 repeats 看方差 |

要点：`contains()` 会把期望短语作为 `facts` 带出，所以"答案里的订单号其实在被压缩掉的历史里"能被识别为 **context** 而不是 model；run 抛异常一律判失败；`completion_rate`（正常结束）与成功率（检查通过）分开统计。**配合回放**：把 `ReplayProvider` 作为评测后端，就得到零成本、确定性的 Harness 回归评测（提示词/工具/压缩/护栏任何漂移都会被归到 `harness`）。

## M6b：MCP
```python
from mini_harness.mcp import McpClient, McpServerParams, register_mcp_tools

async with McpClient("fs", McpServerParams("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/data"])) as c:
    await register_mcp_tools(registry, c, allow={"read_file", "list_directory"})  # 只暴露审过的工具
    agent = Agent(provider, registry, model="...")  # Loop 看不出区别
```
- **最小权限**：权限默认 `EXEC`，仅当服务器标注 `readOnlyHint` 才是 `READ`（才会并行、才会被默认策略放行）。注解来自服务器，不可信；用 `permission_for=` 固定权限，用 `allow=` 白名单。
- **参数校验**：用服务器提供的 JSON Schema 在本地校验，错误以模型可读的 `Invalid arguments: a: 'x' is not of type 'number'` 回填，省一次往返。
- **环境隔离**：子进程只继承最小环境变量（PATH、HOME 等）+ 你显式传的；你的 API Key 不会被意外带给第三方服务器。
- **健壮性**：超时/取消会向服务器发 `notifications/cancelled`；服务器崩溃时所有挂起与后续调用立即失败（不挂死）；单行 16MB 上限；对服务器发来的 `ping` 应答、其他请求回 `-32601`；协议版本不在支持列表则拒绝并清理进程。
- **命名**：`服务器名__工具名`，净化为 `[a-zA-Z0-9_-]` 且 ≤64 字符（超长加稳定哈希后缀）。
- **不可信输入**：工具描述与结果都是不可信文本，会进入模型上下文（提示注入面）；图片/音频只描述不内联。

## 结构
```
mini_harness/
├── core/           messages · loop · context · compaction · tokens · limits · pricing · retry · guards
│                   supervisor · session · events · errors
├── providers/      base · _sse · anthropic · openai_compat · retry · fake
├── session/        models · codec · sqlite_store · manager
├── tools/          spec · registry · executor · policy · builtin
├── observability/  tracer · metrics
├── eval/           recording · replay · checks · attribution · runner
├── mcp/            client · tools
└── sdk.py          Agent 门面（组装根）
examples/           cli.py · eval_demo.py
evals/              真实模型评测套件（16 个任务）+ 运行器 + 使用说明（见 evals/README.md）
tests/              195 个测试（含 kill -9 崩溃恢复、真实 MCP 子进程）
```

## 真实模型评测（evals/）
`uv run python -m evals.run_real --model <id> --repeats 3 --price-in .. --price-out ..`：16 个只读工作区任务，答案由生成的夹具文件**程序化推导**（不硬编码），以 `ANSWER:` 行判分；输出 `summary.md`（含 95% Wilson 区间、基线对比、由真实数字生成的简历表述）、`report.json`、逐次运行与 span。内置 A/B 开关：`--no-cache` / `--no-compaction` / `--no-injection-defense` / `--system-file`。`--dry-run` 用脚本化 oracle 走通整条流水线并**打上 DRY RUN 标记、不生成简历表述**。**这套评测目前只在无模型的情况下自检过；还没有对真实模型跑过。** 用法与可以/不可以宣称什么，见 `evals/README.md`。

自检（`tests/test_real_suite.py`）：夹具确定性、真值与独立重算一致、每个 oracle 轨迹通过且工具真实执行、saboteur（非答案）全部失败、注入被跟随会被判 `unsafe`、context 用例确实触发压缩。这套自检顺带暴露了 3 个真实缺陷（见下）。

## 评测自检发现并已修复的缺陷
| 缺陷 | 影响 | 修复 |
|---|---|---|
| 压缩在"最新一条消息就超阈值"时空转：把阈值前几条小消息折成一条更大的摘要，tokens 反而增加，且每轮重复触发（真实模型下每次都要付一次摘要调用） | 成本与缓存失效 | 旧段太小（< 窗口 5%）不压缩；摘要不比原文小则丢弃并上报花费（`stage="skipped"`） |
| 多提示用例的指标只统计最后一次 run，token/压缩/耗时被低估；run 的成本却是会话累计 | 评测成本与耗时失真 | `combine()` 合并同一会话的全部 run；run span 的成本改为该 run 自身花费 |
| 路径检查器只接受"以期望路径结尾"，把正确的"仅文件名"答案判错 | 假阴性 | 接受完整路径、带前缀的路径或文件名，但不接受别的目录 |

## 与设计文档的对应
| 设计项 | 状态 |
|---|---|
| G1 可控 Loop · G2 工具系统 · G3 双协议 · G4 上下文 · G5 会话恢复 · G6 可观测/评测 | ✅ 全部落地 |
| 沙箱 | L0 进程内 + 权限策略；L1 子进程/rlimit、L2 容器仅预留 `Policy` 接口（ADR-007） |
| 非目标 | 多租户、分布式、向量检索、UI 均未做（按设计） |

## 已知局限（有意保留）
| 项 | 说明 |
|---|---|
| **真实服务未联调** | 与 Anthropic / OpenAI 的交互只用模拟响应验证；MCP 用自带的 stdlib 假服务器验证，未对真实第三方服务器联调 |
| 归因是启发式 | 提供证据与置信度，但"prompt 还是 model"这类边界需 A/B 才能确认；`facts` 需由检查函数声明 |
| MCP 范围 | 仅 stdio、仅 tools；无 HTTP 传输、resources/prompts、sampling、`list_changed` 热刷新；协议版本取 2025-06-18 / 2025-03-26 / 2024-11-05，新版本需自行确认 |
| 回放执行真实工具 | 有副作用的工具会再执行；录制场景请用确定性工具或沙箱 |
| 回退只回退会话状态 | 外部副作用不会撤销 |
| 回合内崩溃 | 丢失进行中的那一轮，继续时会重做；非幂等工具可能重复执行 |
| 估算 token | 无真实分词器：以"上次真实用量+增量估算"校准 |
