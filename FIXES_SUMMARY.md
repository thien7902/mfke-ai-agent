# Bug Fixes Summary - 2026-07-30

## Issue 1: LLM Memory Loss - Tool Execution Not Persisted ✅ FIXED

### Problem
The AI bot had no memory of what tools it executed in previous turns. The LLM would repeat the same tool calls or claim it couldn't do things it had just done.

### Root Cause
**Incomplete conversation history persistence.** After each Holmes call:
1. Only the assistant message (with `tool_calls` metadata) was saved to conversation
2. The actual tool **results** (role: "tool" messages with output text) were never appended
3. On the next turn, Holmes' `_resolve_orphaned_tool_calls()` saw assistant messages with `tool_calls` but no matching "tool" results
4. Holmes treated them as "orphaned/abandoned" and injected fake "Tool execution was cancelled..." results
5. The LLM was actively told its previous tool calls never completed

This violated the OpenAI/Anthropic message format requirement: every assistant `tool_calls` block must be followed by matching "tool" role messages.

### Solution
Modified `holmes_service.py`:

1. **Non-streaming path** (`_convert_llm_result_to_messages`):
   - Changed from returning a single assistant `Message` to returning a list
   - Now returns: `[assistant_msg] + [tool_result_msg_1, tool_result_msg_2, ...]`
   - Each tool result message has `role="tool"`, `tool_call_id`, `name`, and the actual output content
   - Extracts tool output from `LLMResult.tool_calls` dict structure (`result.data`, `result.error`)

2. **Streaming path** (`_stream_chat`):
   - Added accumulators: `accumulated_content`, `tool_calls_accumulated`, `tool_results_accumulated`
   - Captures tool call/result events during streaming
   - After stream ends, constructs and persists assistant + tool result messages to conversation
   - Matches tool_call_ids between calls and results

### Files Changed
- `src/bot/services/holmes_service.py` (lines 505-584, 716-801)

### Testing
After fix, the LLM should:
- Remember what commands it ran and their output
- Not repeat the same tool calls
- Build on previous tool results instead of starting from scratch each turn

---

## Issue 2: Bot Hanging on Bash Commands with Loops ✅ FIXED

### Problem
From user's logs:
```
2026-07-30 00:41:00 - holmes.display.bash_toolset - INFO - Executing bash command: 
kc_path="/root/.kube/active_config" && seed_ctx="kubernetes-admin-sgn10-ops-seed2-private-prd@sgn10-ops-seed2-private-prd" 
&& ns="fke-aiservice-prod-2p94mn6a" && local_port=6443 && svc=$(kubectl --kubeconfig="$kc_path" --context="$seed_ctx" 
get svc -n "$ns" -o name 2>/dev/null | grep -m1 "kube-apiserver" | sed 's|^service/||') && echo "Service: $svc" && 
kill -9 $(lsof -ti ":$local_port" 2>/dev/null) 2>/dev/null || true && nohup kubectl --kubeconfig="$kc_path" 
--context="$seed_ctx" port-forward -n "$ns" "service/$svc" "${local_port}:443" > "/root/.kube/pf-aiservice.log" 2>&1 & 
echo $! > /root/.kube/pf-aiservice.pid && sleep 5 && if kill -0 $(cat /root/.kube/pf-aiservice.pid) 2>/dev/null; 
then echo "Port-forward OK, PID: $(cat /root/.kube/pf-aiservice.pid)"; else echo "Port-forward failed:"; 
cat /root/.kube/pf-aiservice.log; fi
```

Bot then stuck, only health check responding.

### Root Cause
**Incorrect asyncio Future handling in approval callback:**

1. Holmes `bash_toolset.py` detects compound statements (`&&`, `if/then`, `for/while`) in `requires_approval()`
2. Returns `ApprovalRequirement(needs_approval=True)`
3. Calls `approval_callback()` (sync function running in thread pool)
4. Callback creates asyncio Future: `future = loop.create_future()`
5. Schedules `_send_approval_request()` to send Telegram message
6. **BUG**: Tries to wait on future with `asyncio.run_coroutine_threadsafe(future, loop).result(timeout=300)`
   - `run_coroutine_threadsafe()` expects a **coroutine**, not a Future
   - This created a broken wait state that never properly blocked on the future
   - The callback would hang indefinitely or return immediately with wrong result
7. User's button click calls `handle_tool_approval_response()` which sets `future.set_result()`, but the broken wait never sees it

### Solution
Fixed `holmes_service.py:298-308`:

```python
# OLD (BROKEN):
result = asyncio.run_coroutine_threadsafe(future, loop).result(timeout=300)

# NEW (CORRECT):
result = asyncio.run_coroutine_threadsafe(
    asyncio.wait_for(future, timeout=300),
    loop
).result()
```

**Why this works:**
- `asyncio.wait_for(future, timeout=300)` wraps the Future in a coroutine that waits for it with a timeout
- `run_coroutine_threadsafe()` schedules that coroutine on the main event loop
- `.result()` blocks the thread-pool thread until the coroutine completes
- When user clicks approve/deny, `handle_tool_approval_response()` sets the future result
- The coroutine completes and returns the result back to the waiting thread

### Files Changed
- `src/bot/services/holmes_service.py` (lines 298-316)

### Testing
After fix, bash commands with loops/conditionals should:
- Send approval request to Telegram
- Show inline keyboard with Approve/Deny buttons
- Wait for user click
- Continue execution after approval
- Not hang for 5 minutes

---

## Next Steps
1. ✅ Test Issue 1 fix by having a conversation that uses tools across multiple turns
2. ✅ Test Issue 2 fix by running bash commands with loops/conditionals and approving them
3. Restart the bot to apply fixes

## How to Restart the Bot

### If running in Docker:
```bash
docker-compose restart bot
# Or rebuild if you changed dependencies:
docker-compose up -d --build bot
```

### If running directly:
```bash
# Stop the current process (Ctrl+C or kill PID)
# Then restart:
python -m src.bot.main
```

## Verification Steps

### Test Issue 1 (Memory):
1. Start a new conversation in a topic
2. Ask the bot to run a command: "check the kubernetes pods"
3. In the next message, ask: "what did you find?" or "now check the logs for the first pod"
4. The bot should remember what pods it found and reference them
5. ✅ PASS if bot remembers previous tool outputs
6. ❌ FAIL if bot says "I haven't checked yet" or repeats the same command

### Test Issue 2 (Bash approval):
1. Make sure user has TOOL_APPROVAL permission enabled
2. Ask the bot to run a complex command with loops or conditionals
3. Bot should send an approval message with buttons
4. Click "Approve"
5. ✅ PASS if command executes and returns results
6. ❌ FAIL if bot hangs or times out after 5 minutes
