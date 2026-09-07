"""Gateway format adapters (Anthropic↔OpenAI, CrewAI tool JSON)."""

from __future__ import annotations

from ageneval.task.core.openai_compat import (
    anthropic_to_openai_payload,
    coerce_json_object,
    is_anthropic_messages_url,
    openai_to_anthropic_payload,
    rewrite_response_format,
    rewrite_tools_json_schema,
)
from ageneval.task.core.native_tools import (
    canonicalize_tool_args,
    clean_final_answer,
    evidence_from_tool_call,
    is_unusable_final,
    parse_leaked_tool_calls,
    unwrap_tool_kwargs,
)
from ageneval.task.core.result import ToolCall


def test_parse_leaked_crewai_react_and_action():
    leaked = parse_leaked_tool_calls(
        'to=web_search  code:\n{"query":"site:nhs.uk/conditions/shoulder-pain"}\n'
        ' to=web_search code:\n{"query":"NHS Shoulder pain poor balance'
    )
    queries = [c["arguments"]["query"] for c in leaked if c["name"] == "web_search"]
    assert any("nhs.uk/conditions/shoulder-pain" in q for q in queries)
    assert any("Shoulder pain" in q for q in queries)
    action = parse_leaked_tool_calls(
        'Thought: search\nAction: web_search\nAction Input: {"query":"NHS shoulder pain"}'
    )
    assert action == [{"name": "web_search", "arguments": {"query": "NHS shoulder pain"}}]


def test_parse_leaked_tau_and_gdp_tool_names():
    leaked = parse_leaked_tool_calls(
        'Action: find_user_id_by_name_zip\n'
        'Action Input: {"first_name":"Yusuf","last_name":"Rossi","zip":"19122"}',
        allowed_names={"find_user_id_by_name_zip", "get_order_details"},
    )
    assert leaked == [
        {
            "name": "find_user_id_by_name_zip",
            "arguments": {
                "first_name": "Yusuf",
                "last_name": "Rossi",
                "zip": "19122",
            },
        }
    ]
    listed = parse_leaked_tool_calls(
        'to=list_reference_files code:\n{"path":"."}\n',
        allowed_names={"list_reference_files", "read_file"},
    )
    assert listed[0]["name"] == "list_reference_files"
    bare = parse_leaked_tool_calls(
        "Thought: inventory files\nAction: list_reference_files\n",
        allowed_names={"list_reference_files", "read_file"},
    )
    assert bare == [{"name": "list_reference_files", "arguments": {}}]


def test_evidence_reads_open_url_and_search_json():
    page = ToolCall(
        name="open_url",
        arguments={"url": "https://www.nhs.uk/conditions/shoulder-pain/"},
        result={
            "url": "https://www.nhs.uk/conditions/shoulder-pain",
            "text": "Shoulder pain. Hypermobility may cause poor balance and coordination.",
        },
    )
    ev = evidence_from_tool_call(page)
    assert "Hypermobility" in ev
    assert ev.startswith("https://www.nhs.uk/conditions/shoulder-pain")
    search = ToolCall(
        name="web_search",
        arguments={"query": "NHS"},
        result={
            "query": "NHS",
            "results": [
                {
                    "title": "Shoulder pain",
                    "url": "https://www.nhs.uk/conditions/shoulder-pain/",
                    "snippet": "causes",
                }
            ],
        },
    )
    assert "nhs.uk/conditions/shoulder-pain" in evidence_from_tool_call(search)


def test_dsqa_leaked_tool_text_is_unusable():
    react = (
        'to=web_search  code: {"query":"site:nhs.uk/conditions/shoulder-pain"}'
    )
    assert is_unusable_final(react)
    assert clean_final_answer(react) == ""
    assert is_unusable_final("user: None")
    assert clean_final_answer("user: None") == ""
    assert is_unusable_final('{"final_answer":"user: None"}')
    assert clean_final_answer('{"final_answer":"Cervical spondylosis."}') == (
        "Cervical spondylosis."
    )


def test_instructor_tool_calling_schema_is_gateway_safe():
    payload = {
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "call a tool"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "InstructorToolCalling",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "arguments": {
                            "anyOf": [
                                {"type": "object", "additionalProperties": True},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["tool_name", "arguments"],
                },
            },
        },
    }
    out = rewrite_response_format(payload)
    js = out["response_format"]["json_schema"]
    assert js["strict"] is False
    obj = js["schema"]["properties"]["arguments"]["anyOf"][0]
    assert obj["additionalProperties"] is False
    tools = rewrite_tools_json_schema(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_reference_files",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        }
    )
    assert tools["tools"][0]["function"]["parameters"]["additionalProperties"] is False


def test_coerce_list_of_objects_takes_first():
    assert coerce_json_object(
        [{"product_id": "1656367028"}, {"product_id": "1656367028"}]
    ) == {"product_id": "1656367028"}
    assert coerce_json_object(
        '[{"product_id": "1"}, {"product_id": "2"}]'
    ) == {"product_id": "1"}


def test_canonicalize_tau_name_zip_aliases():
    assert canonicalize_tool_args(
        "find_user_id_by_name_zip",
        {"name": "Yara Silva", "zip_code": "77159"},
    ) == {"first_name": "Yara", "last_name": "Silva", "zip": "77159"}
    assert canonicalize_tool_args(
        "find_user_id_by_name_zip",
        {"full_name": "Yara Silva", "zip_code": "77159"},
    ) == {"first_name": "Yara", "last_name": "Silva", "zip": "77159"}


def test_parse_leaked_leftover_action_json():
    leftover = (
        '{"action":"code_exec","arguments":{"code":"d={1:2}; print(d)"}}'
    )
    leaked = parse_leaked_tool_calls(
        leftover, allowed_names={"code_exec", "finish"}
    )
    assert leaked[0]["name"] == "code_exec"
    assert "print(d)" in leaked[0]["arguments"]["code"]
    assert is_unusable_final(leftover)
    assert clean_final_answer(leftover) == ""
    addr = (
        '{"action":"modify_pending_order_address","arguments":'
        '{"order_id":"#W3730488","address":{"city":"New York"}}}'
    )
    leaked2 = parse_leaked_tool_calls(
        addr, allowed_names={"modify_pending_order_address"}
    )
    assert leaked2[0]["name"] == "modify_pending_order_address"
    assert leaked2[0]["arguments"]["order_id"] == "#W3730488"
    messy = 'to=code_exec 代 наңjson\n{"code":"print(3)"}'
    leaked3 = parse_leaked_tool_calls(messy, allowed_names={"code_exec"})
    assert leaked3[0]["name"] == "code_exec"
    assert leaked3[0]["arguments"]["code"] == "print(3)"
    assert is_unusable_final(messy)


def test_unwrap_nested_action_arguments():
    assert unwrap_tool_kwargs(
        {"action": "code_exec", "arguments": {"code": "print(1)"}}
    ) == {"code": "print(1)"}


def test_unwrap_list_scalar_and_dict():
    assert unwrap_tool_kwargs({"product_id": ["1656367028", "4896585277"]}) == {
        "product_id": "1656367028"
    }
    assert unwrap_tool_kwargs(
        {"arguments": [{"product_id": "1656367028"}]}
    ) == {"product_id": "1656367028"}


def test_messages_url_only_on_gateway():
    assert is_anthropic_messages_url("http://10.12.111.133:49183/v1/messages")
    assert not is_anthropic_messages_url("https://api.anthropic.com/v1/messages")
    assert not is_anthropic_messages_url("http://host/v1/chat/completions")


def test_anthropic_to_openai_tools_and_max_tokens():
    payload = {
        "model": "gpt-5.6-sol",
        "max_tokens": 256,
        "system": "You are helpful.",
        "tools": [
            {
                "name": "web_search",
                "description": "search",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
        "messages": [{"role": "user", "content": "What is NHS?"}],
    }
    out = anthropic_to_openai_payload(payload)
    assert out["model"] == "gpt-5.6-sol"
    assert out.get("max_completion_tokens") == 256
    assert "max_tokens" not in out
    assert out["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert out["messages"][1]["content"] == "What is NHS?"
    assert out["tools"][0]["function"]["name"] == "web_search"
    assert out["tools"][0]["function"]["parameters"]["required"] == ["query"]


def test_anthropic_tool_result_roundtrip_roles():
    payload = {
        "model": "gpt-5.6-sol",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "web_search",
                        "input": {"query": "NHS"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": '{"results":[]}',
                    }
                ],
            },
        ],
    }
    out = anthropic_to_openai_payload(payload)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["assistant", "tool"]
    assert out["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert out["messages"][1]["tool_call_id"] == "call_1"


def test_openai_tool_calls_become_anthropic_blocks():
    payload = {
        "id": "resp_1",
        "model": "gpt-5.6-sol",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"NHS spondylosis"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }
    out = openai_to_anthropic_payload(payload)
    assert out["type"] == "message"
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0]["type"] == "tool_use"
    assert out["content"][0]["name"] == "web_search"
    assert out["content"][0]["input"] == {"query": "NHS spondylosis"}
    assert out["usage"]["input_tokens"] == 10
