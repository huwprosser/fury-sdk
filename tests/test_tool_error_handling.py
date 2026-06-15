"""
Test that tool errors are properly handled and don't cause infinite loops.
This test ensures that when a tool raises an exception, the agent sees the error
and doesn't keep retrying with different parameters.
"""
import asyncio
import pytest
from fury.tools import ToolRegistry, ToolExecutor
from fury.types import ChatStreamEvent, ToolCallEvent


def failing_tool():
    """A tool that always raises an exception."""
    raise FileNotFoundError("This tool always fails")


def test_tool_error_handling():
    """Test that tool errors are properly captured and returned."""
    
    registry = ToolRegistry()
    registry.register_builtin(
        name="fail_tool",
        description="A failing tool",
        parameters={},
        func=failing_tool,
    )
    
    executor = ToolExecutor(registry)
    
    tool_calls = [{
        "id": "test123",
        "function": {
            "name": "fail_tool",
            "arguments": "{}"
        }
    }]
    
    active_history = []
    session = None
    
    def run_test():
        events = []
        async def collect_events():
            async for event in executor.execute_tool_calls(tool_calls, active_history, session):
                events.append(event)
        
        asyncio.run(collect_events())
        return events, active_history
    
    events, active_history = run_test()
    
    tool_events = [event.tool_call for event in events if event.tool_call]
    history_events = [event.history_delta for event in events if event.history_delta]

    assert len(tool_events) == 2
    assert tool_events[0].id == "test123"
    assert tool_events[0].status == "started"
    assert tool_events[0].tool_name == "fail_tool"
    assert tool_events[0].arguments == {}
    assert tool_events[0].result is None

    assert tool_events[1].id == "test123"
    assert tool_events[1].status == "error"
    assert tool_events[1].tool_name == "fail_tool"
    assert tool_events[1].arguments is None
    assert tool_events[1].result is not None
    assert "Error" in tool_events[1].result
    assert "This tool always fails" in tool_events[1].result

    assert len(history_events) == 1
    assert history_events[0].kind == "tool_result"

    # Active history should contain the tool result
    assert len(active_history) == 1
    assert active_history[0]["role"] == "tool"
    assert active_history[0]["name"] == "fail_tool"
    assert active_history[0]["tool_call_id"] == "test123"
    assert active_history[0]["content"] == tool_events[1].result


def test_tool_error_prevents_infinite_loop():
    """Test that tool errors prevent the agent from looping."""
    
    call_count = 0
    
    def counting_failing_tool():
        nonlocal call_count
        call_count += 1
        raise ValueError(f"Tool failed on attempt {call_count}")
    
    registry = ToolRegistry()
    registry.register_builtin(
        name="counting_fail_tool",
        description="A failing tool that counts calls",
        parameters={},
        func=counting_failing_tool,
    )
    
    executor = ToolExecutor(registry)
    
    # Simulate multiple rounds of tool calling (like an agent would do)
    for round_num in range(5):
        call_count = 0
        tool_calls = [{
            "id": f"test_{round_num}",
            "function": {
                "name": "counting_fail_tool",
                "arguments": "{}"
            }
        }]
        
        active_history = []
        session = None
        
        def run_test():
            async def collect_events():
                async for event in executor.execute_tool_calls(tool_calls, active_history, session):
                    if event.tool_call and event.tool_call.result:
                        # The error should be in the result
                        assert "Error" in str(event.tool_call.result)
            
            asyncio.run(collect_events())
        
        run_test()
        
        # Tool should only be called once per round, not in a loop
        assert call_count == 1, f"Tool was called {call_count} times in round {round_num}"


if __name__ == "__main__":
    print("Running test_tool_error_handling...")
    test_tool_error_handling()
    print("✅ test_tool_error_handling passed")
    
    print("\nRunning test_tool_error_prevents_infinite_loop...")
    test_tool_error_prevents_infinite_loop()
    print("✅ test_tool_error_prevents_infinite_loop passed")
    
    print("\n🎉 All tests passed!")
