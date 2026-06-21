# HistoryCompactor

`HistoryCompactor` summarizes a list of chat history messages into a single string using a Fury `Agent`.

```python
from fury import Agent, HistoryCompactor

agent = Agent(
    model="your-summary-model",
    system_prompt="You summarize chat history.",
)

compactor = HistoryCompactor(agent)
summary = await compactor.compact(history)
```

You can override the prompt either at construction time or per call:

```python
compactor = HistoryCompactor(agent, prompt="Summarize these messages.")
summary = await compactor.compaction_summary(history, prompt="Custom prompt")
```

Default prompt:

```text
Summarize this chat for future turns.
Keep the summary concise but preserve:
- the user's goals and preferences
- important decisions and constraints
- facts that would be needed to continue the conversation
- projects, plans, promises, and unresolved questions
- relevant files, URLs, or tool results

Do not include filler. Do not mention that this is an automatic compaction.
```
