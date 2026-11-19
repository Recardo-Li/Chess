"""Run one API-only LLM versus Lichess AI game."""

from __future__ import annotations

import argparse
import getpass
import math
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import yaml

from chess_eval.chess_environment import ChessEnvironment
from chess_eval.lichess_client import LichessClient, LichessError
from chess_eval.llm_client import LLMClient
from chess_eval.runner import GameRunner

ROOT = Path(__file__).resolve().parent


def read_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except yaml.YAMLError:
        # YAML errors can include the offending line, potentially a token.
        raise ValueError(f"{path.name} 的 YAML 语法无效，请检查缩进。") from None
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} 必须是 YAML 字典。")
    return value


def integer(config: dict, key: str, lower: int, upper: int) -> None:
    value = config.get(key)
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"配置项 {key} 必须是 {lower}–{upper} 的整数。")


def load_config(root: Path = ROOT) -> tuple[dict, dict, str]:
    lichess = read_yaml(root / "Conifg" / "Lichess.yaml")
    llm = read_yaml(root / "Conifg" / "LLM.yaml")
    allowed_lichess = {
        "api_token", "base_url", "level", "color", "clock_limit", "clock_increment",
        "days", "request_timeout", "stream_timeout", "reconnect_attempts",
    }
    allowed_llm = {
        "provider", "model", "base_url", "request_timeout", "invalid_move_retries",
        "include_legal_moves", "constrain_legal_moves", "request_options", "skill_path", "output_dir", "max_plies",
    }
    if set(lichess) - allowed_lichess or set(llm) - allowed_llm:
        raise ValueError("配置中有未支持的字段；LLM key 请通过命令行或环境变量提供，不要写入 YAML。")
    integer(lichess, "level", 1, 8)
    integer(lichess, "reconnect_attempts", 0, 20)
    if lichess.get("color") not in ("white", "black", "random"):
        raise ValueError("Lichess.color 必须是 white、black 或 random（指 LLM 执棋颜色）。")
    if lichess.get("clock_limit") is None:
        integer(lichess, "days", 1, 14)
        if lichess["days"] not in (1, 2, 3, 5, 7, 10, 14):
            raise ValueError("Lichess.days 只能是 1、2、3、5、7、10、14。")
    else:
        integer(lichess, "clock_limit", 0, 10800)
        integer(lichess, "clock_increment", 0, 60)
        if lichess["clock_limit"] < 60:
            raise ValueError("BOT 不支持 UltraBullet；本程序要求 clock_limit 至少 60 秒。")
    for key in ("request_timeout", "stream_timeout"):
        value = lichess.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Lichess.{key} 必须为正数。")
    if not isinstance(lichess.get("api_token"), str):
        raise ValueError("Lichess.api_token 必须是字符串；尚未配置时请保留空引号。")
    url = urlsplit(str(lichess.get("base_url", "")))
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Lichess.base_url 必须是不含凭证及查询参数的 HTTPS 地址。")
    integer(llm, "max_plies", 1, 10000)
    retries = llm.get("invalid_move_retries", 0)
    if type(retries) is not int or retries < 0:
        raise ValueError("LLM.invalid_move_retries 必须为非负整数。")
    for key in ("include_legal_moves", "constrain_legal_moves"):
        if type(llm.get(key, False)) is not bool:
            raise ValueError(f"LLM.{key} 必须为 true 或 false。")
    if llm.get("constrain_legal_moves") and not llm.get("include_legal_moves"):
        raise ValueError("启用 constrain_legal_moves 时必须同时设置 include_legal_moves: true。")
    for key in ("skill_path", "output_dir"):
        if not isinstance(llm.get(key), str) or not llm[key].strip():
            raise ValueError(f"LLM.{key} 必须是非空路径。")
    skill_text = (root / llm["skill_path"]).read_text(encoding="utf-8-sig")
    # Validate provider/options without network requests or real credentials.
    validator = LLMClient(llm, "configuration-validation", skill_text,
                          require_constraints=llm.get("constrain_legal_moves", False))
    validator.close()
    return lichess, llm, skill_text


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="通过 API 运行一局 LLM vs Lichess AI")
    parser.add_argument("--api-key", help="被测 LLM 的 key；也可使用环境变量或启动时隐藏输入")
    parser.add_argument("--check-config", action="store_true", help="离线检查配置，不调用 API，不要求填写 key")
    args = parser.parse_args()
    lichess = llm = None
    try:
        lichess_config, llm_config, skill_text = load_config(ROOT)
        if args.check_config:
            print("配置和棋规 skill 检查通过（未验证远程凭证、模型权限或账号类型）。")
            return 0
        if not lichess_config["api_token"].strip():
            raise ValueError("请先在 Conifg/Lichess.yaml 中填写 api_token。")
        env_name = "OPENAI_API_KEY" if llm_config["provider"] == "gpt" else "DEEPSEEK_API_KEY"
        api_key = args.api_key or os.environ.get(env_name)
        if not api_key:
            if not sys.stdin.isatty():
                raise ValueError(f"缺少 LLM key：请设置 {env_name} 或使用 --api-key。")
            api_key = getpass.getpass(f"输入 {llm_config['provider']} API key（不会显示或保存）: ")
        strict = llm_config.get("constrain_legal_moves", False)
        llm = LLMClient(llm_config, api_key, skill_text, require_constraints=strict)
        lichess = LichessClient(lichess_config)
        environment = ChessEnvironment(
            lichess, lichess_config, max_plies=llm_config["max_plies"],
            include_legal_moves=llm_config.get("include_legal_moves", False),
            constrain_legal_moves=strict,
        )
        runner = GameRunner(
            environment, llm, output_dir=ROOT / llm_config["output_dir"],
            skill_text=skill_text, agent_name=llm_config["model"],
            settings={"llm": llm_config, "lichess": {k: v for k, v in lichess_config.items() if k != "api_token"}},
            invalid_action_retries=llm_config.get("invalid_move_retries", 0),
        )
        summary = runner.run()
        print(f"评测状态：{summary['outcome']}；棋谱结果：{summary['result']}")
        print(f"记录目录：{runner.output_dir}")
        return 1 if summary["outcome"] in {"error", "output_error", "interrupted"} else 0
    except (ValueError, OSError, LichessError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("已取消。", file=sys.stderr)
        return 130
    finally:
        if lichess is not None:
            lichess.close()
        if llm is not None:
            llm.close()


if __name__ == "__main__":
    raise SystemExit(main())
