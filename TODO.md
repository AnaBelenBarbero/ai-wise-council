async def call_model(state: MessagesState):
    # Deduplicate messages
    state["messages"] = deduplicate_messages(state["messages"])
    
    # Summarize if conversation is long
    if len(state["messages"]) > 10:
        state["messages"] = await summarize_history(state["messages"], model)
    
    # Only trim if still needed after summarization
    if calculate_tokens(state["messages"]) > MAX_TOKENS:
        messages = trimmer.invoke(state["messages"])
    else:
        messages = state["messages"]
    
    prompt = prompt_template.invoke(
        {"messages": messages, "language": state.get("language", "English")},
        config
    )
    
    response = await call_model_with_retry(prompt)
    return {"messages": [response]}

# Update your workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())





7. **Implement message summarization** for long conversations:

```python:notebook.ipynb
async def summarize_history(messages: list, model) -> str:
    """Summarize conversation history to reduce token usage"""
    if len(messages) <= 3:  # Don't summarize short conversations
        return messages
        
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the key points of this conversation concisely."),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    summary = await model.ainvoke(summary_prompt.invoke({"messages": messages}))
    
    # Return the summary as a system message plus the last exchange
    return [
        SystemMessage(content=summary.content),
        messages[-2],  # Last human message
        messages[-1]   # Last assistant message
    ]
```
