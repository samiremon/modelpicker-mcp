"""Exercise the agent loop with real Gemini types and a real MCP server, but no network."""

import asyncio
import json

import pytest
from google.genai import types
from mcp import Client

from agents import test_agent
from agents.connection import EXIT_PROVIDER_ERROR, EXIT_STEP_LIMIT, bait_server_params
from agents.test_agent import ProviderError, StepLimitReached, run_agent
from scanner.attacks import load_attack


def reply(*parts, finish=types.FinishReason.STOP):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)), finish_reason=finish
            )
        ]
    )


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


@pytest.fixture
def bait_env(tmp_path, monkeypatch):
    attack_file = tmp_path / "attack.json"
    log = tmp_path / "calls.jsonl"
    attack_file.write_text(json.dumps(load_attack("email_exfiltration_01")))
    monkeypatch.setenv("BAIT_ATTACK_FILE", str(attack_file))
    monkeypatch.setenv("BAIT_CALL_LOG", str(log))
    return log


def play(replies):
    seen = []

    async def generate(contents, tools):
        seen.append({"contents": list(contents), "tools": tools})
        return replies.pop(0) if replies else reply(call("read_email"))

    async def go():
        async with Client(bait_server_params()) as client:
            try:
                return await run_agent(client, "Summarize my latest email.", generate), None
            except (ProviderError, StepLimitReached) as error:
                return None, error

    answer, error = asyncio.run(go())
    if error:
        raise error
    return answer, seen


def test_loop_returns_tool_output_to_the_model(bait_env):
    answer, seen = play([reply(call("read_email")), reply(types.Part(text="Moved to 3 PM."))])
    assert answer == "Moved to 3 PM."
    declared = {d.name for d in seen[0]["tools"][0].function_declarations}
    assert declared == {"read_email", "get_config", "send_email"}
    returned = seen[1]["contents"][-1].parts[0].function_response
    assert returned.name == "read_email" and "attacker@evil.test" in returned.response["result"]
    assert json.loads(bait_env.read_text().splitlines()[0])["tool"] == "read_email"


def test_safety_block_is_a_provider_error(bait_env):
    with pytest.raises(ProviderError):
        play([types.GenerateContentResponse(candidates=[])])


def test_early_stop_is_a_provider_error(bait_env):
    with pytest.raises(ProviderError):
        play([reply(types.Part(text=""), finish=types.FinishReason.SAFETY)])


def test_endless_tool_calls_hit_the_step_limit(bait_env):
    with pytest.raises(StepLimitReached):
        play([])


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        ([reply(types.Part(text="Moved to 3 PM."))], 0),
        ([types.GenerateContentResponse(candidates=[])], EXIT_PROVIDER_ERROR),
        ([], EXIT_STEP_LIMIT),
    ],
)
def test_main_turns_failures_into_exit_codes(bait_env, monkeypatch, replies, expected):
    async def generate(contents, tools):
        return replies.pop(0) if replies else reply(call("read_email"))

    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("AGENT_TASK", "Summarize my latest email.")
    monkeypatch.setattr(test_agent.genai, "Client", lambda: None)
    monkeypatch.setattr(test_agent, "gemini_generate", lambda client: generate)
    assert asyncio.run(test_agent.main()) == expected


def test_loop_sees_a_tool_description_change_mid_session(tmp_path, monkeypatch):
    attack_file = tmp_path / "attack.json"
    log = tmp_path / "calls.jsonl"
    attack_file.write_text(json.dumps(load_attack("rug_pull_calendar_01")))
    monkeypatch.setenv("BAIT_ATTACK_FILE", str(attack_file))
    monkeypatch.setenv("BAIT_CALL_LOG", str(log))

    _, seen = play([reply(call("get_calendar")), reply(types.Part(text="2 PM"))])

    def description(step):
        declarations = seen[step]["tools"][0].function_declarations
        return next(d.description for d in declarations if d.name == "get_calendar")

    assert "audit_log" not in description(0)
    assert "audit_log" in description(1)
    assert "__tools_changed__" in log.read_text()
