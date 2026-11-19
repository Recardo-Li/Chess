# LLM 国际象棋评测

一个最小的单局循环：Lichess 推送局面 → 将棋规 skill、FEN、棋盘和历史走法发送给 LLM → 本地校验 UCI 走法 → 通过 Lichess Bot API 落子 → 等待内置 AI → 重复直到终局。

LLM 支持 GPT 和 DeepSeek，双方都通过 HTTP API 调用；不需要浏览器或本地 Stockfish。`python-chess` 仅维护棋盘、校验规则和导出棋谱，不负责选择走法。配置目录按需求命名为 **Conifg**。

## Agent 和环境如何协作

| 组件 | 职责 |
| --- | --- |
| `ChessEnvironment` | 从 Lichess 同步对局，轮到被测方时生成观察。它校验动作、确认双方落子、处理重连和终局，并导出 PGN。 |
| `LLMClient` | 实现 `act(observation, feedback)`。每次调用结合任务 skill 和当前观察，请求 GPT 或 DeepSeek 一次，返回动作字符串和调用统计。 |
| `GameRunner` | 从环境取得回合，调用 Agent，将动作交给环境校验。校验失败时按配置把原因反馈给 Agent，在同一局面重试；通过后提交一次，并记录评测结果。 |

共用接口在 `chess_eval/contracts.py`。一份 `Observation` 包含局面编号、给模型看的文本和输出格式，可附带严格约束用的候选值。国际象棋环境把 FEN、棋盘图和双方走法历史写入文本。提示开关与输出约束的组合如下。

| `include_legal_moves` | `constrain_legal_moves` | 模型收到的输入与输出方式 |
| --- | --- | --- |
| `false` | `false` | 只给局面和任务说明，自行生成 UCI。 |
| `true` | `false` | 提示中列出全部合法走法，自由文本返回 UCI。 |
| `true` | `true` | 提示中列出全部合法走法，同一份候选列表也用于 API 的 `enum`。 |

三种模式下，环境都会按当前局面检查动作。终局由环境直接报告，循环随之结束。

要接入另一种游戏，实现 `Environment` 的 `turns`、`validate`、`submit` 和 `finish`，在启动处传给同一个 `GameRunner`，为模型提供该游戏的 skill。要接入另一个 Agent，实现 `act` 并返回 `Decision` 即可。`tests/test_environment_contract.py` 用一个取筹码任务演示了同一环境接入 LLM 和本地策略两种方式。日常运行国际象棋评测仍使用 `python run.py` 和原有两份 YAML。

## 1. 克隆后启动

**Windows 10/11 x64** 的解释器、项目依赖和 HTTPS 证书随仓库一起提供，克隆后可直接用。首次配置时，在项目目录用 PowerShell 复制 Lichess 模板，并在新文件的 `api_token` 中填写自己的 token。已有配置可以跳过复制。

```powershell
Copy-Item Conifg/Lichess.example.yaml Conifg/Lichess.yaml
```

选好 `Conifg/LLM.yaml` 中的模型后，按终端情况启动。两种方式都会使用本仓库的解释器，并在需要时提示输入 LLM key。

| 新电脑上的终端 | 开始对局 | 离线检查配置 |
| --- | --- | --- |
| 已能运行 `python`（Python 3.10+） | `python run.py` | `python run.py --check-config` |
| 未安装 Python，或输入 `python` 打开了 Microsoft Store | `.\run.cmd` | `.\run.cmd --check-config` |

如果新电脑没有系统 Python，但希望在当前 PowerShell 里使用 `python run.py`，可以先把随仓库提供的解释器加入这个终端的搜索路径。

```powershell
$env:Path = (Resolve-Path .\.venv\Scripts).Path + ';' + $env:Path
python run.py
```

Windows 的 `.venv/Scripts/python.exe` 来自 Python 官方嵌入式发行版，依赖位于 `.venv/Lib/site-packages`，导入路径相对于仓库解析。PowerShell 通过 `PATH` 查找裸命令 `python`，所以没有该命令的电脑使用仓库内的 `.\run.cmd`。[Python Windows 发行包](https://www.python.org/downloads/release/python-31410/) · [PowerShell 命令查找规则](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_command_precedence)

**macOS 和 Linux** 使用本机 Python 3.10+。在自己的虚拟环境中安装依赖，再按上面的模板准备 `Conifg/Lichess.yaml`。

```powershell
python -m pip install -r requirements.txt
python run.py --check-config
```

## 2. 准备 Lichess BOT 账号

必须使用专门的 **BOT 账号**。Lichess Bot API 允许程序下棋；Board API 不允许引擎辅助，因此这里使用 Bot API。[官方 API 文档](https://lichess.org/api#tag/Bot)

1. 准备一个**从未下过棋的新 Lichess 账号**。
2. 登录该账号，在 [token 页面](https://lichess.org/account/oauth/token) 创建 token，勾选 `bot:play`（Play games with the bot API）和 `challenge:write`（Create, accept, decline challenges）。
3. 将 token 写入 `Conifg/Lichess.yaml` 的 `api_token`。
4. 如果账号尚未升级为 BOT，手动执行一次下面的命令。**升级不可逆**；不要对日常使用的账号操作。程序本身仅检查 BOT 身份，不会自动升级。[官方升级说明](https://lichess.org/api#tag/Bot/operation/botAccountUpgrade)

```powershell
.\.venv\Scripts\python.exe -c "import requests,yaml; c=yaml.safe_load(open('Conifg/Lichess.yaml',encoding='utf-8')); r=requests.post(c['base_url'].rstrip('/')+'/api/bot/account/upgrade',headers={'Authorization':'Bearer '+c['api_token']},timeout=30); print('HTTP',r.status_code)"
```

HTTP 200 表示成功；若不是 200，请查看官方升级条件及 token 权限。不要重复运行升级命令来解决登录问题。

Lichess 配置示例：

```yaml
api_token: "填写你的Lichess-token"
base_url: "https://lichess.org"
level: 1
color: "white"
clock_limit: null
clock_increment: 0
days: 1
request_timeout: 30
stream_timeout: 90
reconnect_attempts: 3
```

`level` 为 1–8；`color` 是 **LLM** 的颜色，可选 `white`、`black`、`random`。默认通信棋每步 1 天，适合有推理延迟的 API。改成 `clock_limit: 1800`、`clock_increment: 30` 即为 30 分钟、每步加 30 秒，此时忽略 `days`。计时棋中的模型/API 延迟会消耗棋钟。[AI 挑战接口](https://lichess.org/api#tag/Challenges/operation/challengeAi)

## 3. 配置 LLM 和 key

修改 `Conifg/LLM.yaml`。GPT 默认：

```yaml
provider: "gpt"
model: "gpt-5.4-mini"
base_url: "https://api.openai.com/v1"
```

DeepSeek 切换为：

```yaml
provider: "deepseek"
model: "deepseek-flash"
base_url: "https://api.deepseek.com"
```

保留文件中的其他字段。`model` 可替换为账号能访问且支持 Chat Completions 的模型；程序不维护固定模型白名单。[GPT 默认模型](https://developers.openai.com/api/docs/models/gpt-5.4-mini)、[DeepSeek 接口说明](https://api-docs.deepseek.com/)

LLM key **不写入配置文件**。最简单的启动方式：

```powershell
python run.py
```

程序在终端提示输入 LLM key，输入不会显示，也不会持久化。也支持一次性参数：

```powershell
python run.py --api-key "你的LLM-key"
```

或者先在终端设置环境变量，再使用 `python run.py`：

```powershell
# GPT
$env:OPENAI_API_KEY = "你的GPT-key"
# 使用 DeepSeek 时设置此变量
$env:DEEPSEEK_API_KEY = "你的DeepSeek-key"
python run.py
```

读取优先级为 `--api-key` → 当前 provider 对应的环境变量 → 隐藏输入。参数及明文环境变量赋值可能留在终端历史中，隐藏输入可避免这一点。程序不会记录请求头或 key；`Conifg/Lichess.yaml` 已加入 `.gitignore`。

## 4. 评测设置和结果

- `skill_path`：基础棋规，默认 `skills/chess-rules/SKILL.md`；每次调用作为 system 指令加载。
- `include_legal_moves: false`：默认不提供候选走法，模型自行判断合法性。开启后可更侧重候选走法选择能力；跨模型比较时保持一致。
- `constrain_legal_moves: true`：启用 API 的严格输出约束，必须同时设置 `include_legal_moves: true`。本项目当前配置已开启；省略这个字段或设为 `false` 时保留原先的纯文本 UCI 模式。
- `invalid_move_retries: 0`：默认首次非法输出即停止评测，记为 `invalid_move`、得分 0，并向 Lichess 请求认输。设为 N 时最多额外纠错 N 次，所有响应都记录。
- `max_plies: 300`：最多 300 个半回合（一方走一步算一个）。达到上限记为 `ply_limit`，得分为空，不当作和棋。
- `request_timeout`：单次 LLM 请求的超时秒数。
- `request_options`：可填写 `temperature`、`top_p`、`max_completion_tokens`、`max_tokens`、`reasoning_effort`、`thinking`、`seed`、`stop`。具体支持情况取决于模型；两种 token 上限只能填一种，推理模型需留出足够的推理 token。默认 `{}` 使用服务端默认参数。

纯文本模式只接受一条完整的 UCI 走法，例如 `e2e4`、`e1g1`、`a7a8q`；约束模式接受下文规定的 JSON 或工具参数。程序不会把非法输出自动改写为另一条走法。模型请求以当前局面和完整棋谱为上下文，不附带前几回合的模型思考内容。

### 合法走法输出约束

启用严格约束时，环境用 `python-chess` 生成当前一方的全部合法走法，模型适配器用这些候选值构造以下 schema（`enum` 中实际填入完整列表）：

```json
{
  "type": "object",
  "properties": {"move": {"type": "string", "enum": ["e2e4", "d2d4"]}},
  "required": ["move"],
  "additionalProperties": false
}
```

GPT 使用 `response_format.type: json_schema` 和 `strict: true`，解析 `{"move":"e2e4"}`。所选模型和 API 服务必须支持 Structured Outputs。[OpenAI 官方说明](https://developers.openai.com/api/docs/guides/structured-outputs)

DeepSeek 使用 Beta 的 `strict: true` 工具调用，将相同 schema 作为 `choose_move` 的参数约束。`choose_move` 只表示模型的走法选择，实际落子仍在本地校验之后进行。需配置：

```yaml
base_url: "https://api.deepseek.com/beta"
include_legal_moves: true
constrain_legal_moves: true
```

思考模式、推理强度和 token 预算继续由原有 `request_options` 决定，程序不会为启用约束而自动关闭思考。DeepSeek 严格模式支持思考与非思考两种模式，但强制指定工具调用只支持非思考模式。因此：

- 非思考模式：请求强制调用一次 `choose_move`。
- 思考模式：使用 `tool_choice: auto`，提示模型调用 `choose_move`；只接受恰好一个符合 schema 的调用。模型可能只返回文本或一直思考，程序会明确报告输出失败，不能承诺每次都产生走法。

[DeepSeek 严格模式说明](https://api-docs.deepseek.com/guides/tool_calls/)、[工具选择限制](https://api-docs.deepseek.com/api/create-chat-completion/)

严格模式的保证针对正常完成的结构化输出。空回复、截断、拒答、未调用工具或接口不支持等情况仍可能发生。程序始终再次验证结构和棋盘合法性；不符合约束就停止，不降级为自由文本，也不偷偷选择候选列表中的第一步。这些输出失败不使用 `invalid_move_retries` 自动重试。

约束开启后，评测主要考查合法候选中的走法选择能力，不能再用结果衡量模型独立生成合法走法的能力。比较模型时应保持约束开关、思考模式、预算及重试设置一致。

每局生成 `outputs/<UTC时间>-<随机后缀>/`：

| 文件 | 内容 |
| --- | --- |
| `game.pgn` | 服务端确认过的棋谱，可导入棋谱工具 |
| `moves.jsonl` | 每次模型最终响应或工具参数、合法性、耗时、token 用量、`finish_reason`、`output_mode`，以及落子确认；不保存推理正文 |
| `summary.json` | 胜负/得分、合法响应率、输出失败次数、token 总量、配置快照（移除 token）、终止原因 |
| `skill.md` | 本局实际使用的棋规快照 |

自然终局按 Lichess 判定：胜 1、和 0.5、负 0。网络/API 错误记为 `error`，中断记为 `interrupted`，这些不计作模型棋力失败。`result` 是最后确认的棋谱结果；提前结束和认输请求尚未同步时保留 `*`，不会伪造服务端终局。

空回复、`finish_reason: length` 截断、拒答和约束协议异常记为 `output_error`，得分为空；每次尝试的 `error` 会进一步标明 `empty_response`、`output_truncated`、`output_refused`、`missing_tool_call` 或 `constraint_violation` 等原因。`output_error_responses` 单独统计这些失败，`legal_response_rate` 的分母不包含它们，比较模型时需要同时查看。纯文本模式中的真实非法走法或 UCI 格式错误仍记为 `invalid_move`。旧运行记录不会被重写。

断线会在同一局中重新同步，最多 `reconnect_attempts` 次；落子请求结果不确定时先检查服务端棋谱，不会重复调用模型或盲目重发。若仍无法确认则停止，记录 `error`。API 429 不立即重试，Lichess 限流后应至少等待一分钟。

Ctrl+C、非法走法、调用失败或步数上限会请求结束本局并保存记录。如果打印“认输请求未确认”，请打开本局链接检查。创建对局请求超时时可能已经创建成功，但程序拿不到 id；请检查账号的当前对局后再运行。这个最小版本每次运行创建一局，不支持恢复历史运行。

## 本地验证

Windows 下可以直接使用打包的运行环境。

```powershell
.\run.cmd --check-config
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用模拟 API，不消耗 LLM 额度，也不会创建真实 Lichess 对局。`--check-config` 只检查本地格式和规则文件，不验证账号权限、余额、模型可用性或网络。真实端到端对战需要你填入两家的凭证后运行。
