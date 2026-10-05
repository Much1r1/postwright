# Sample Post Example 1 (Build Log)

Spent 3 hours debugging why my LangGraph checkpointer was dropping state during interrupt resume.

Turns out `db.sqlite3` connection wasn't persisting thread-local transactions across CLI re-invocations.

Fixed by switching to thread-scoped connection pooling with explicitly defined thread_id checkpointer.

Lesson: Checkpoint persistence in HITL workflows must be tested across process restarts, not just in-memory.
