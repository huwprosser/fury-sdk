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
    
    # Should have exactly 2 events:
    # 1. Tool call with arguments (result=None)
    # 2. Tool call with result (the error message)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"
    
    # First event should be the tool call with arguments
    assert events[0].tool_call is not None
    assert events[0].tool_call.tool_name == "fail_tool"
    assert events[0].tool_call.arguments == {}
    assert events[0].tool_call.result is None
    
    # Second event should be the tool call with the error result
    assert events[1].tool_call is not None
    assert events[1].tool_call.tool_name == "fail_tool"
    assert events[1].tool_call.arguments is None
    assert events[1].tool_call.result is not None
    assert "Error" in events[1].tool_call.result
    # Error message format is "Error: <exception message>"
    assert "This tool always fails" in events[1].tool_call.result
    
    # Active history should contain the tool result
    assert len(active_history) == 1
    assert active_history[0]["role"] == "tool"
    assert active_history[0]["name"] == "fail_tool"
    assert active_history[0]["content"] == events[1].tool_call.result


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
