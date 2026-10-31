"""Fork (branch) a session: a fresh session that inherits the active transcript of its parent.

Unlike a rewind (``/undo``, which deactivates rows in place and leaves one live path) a fork keeps
the parent byte-identical and gives the child its own rows, so both conversations stay resumable —
the shape the local/ACP branch verb already uses (``hermes_state_mutation_branch``).

The child is published with ``_branched_from`` in its ``model_config``: the same marker that tells
``_publish_child_session_row`` this is a new conversation (unpinned, visible) rather than a
compression continuation. Everything runs inside ONE ``BEGIN IMMEDIATE`` transaction so a live
gateway cannot append between the boundary read and the row clone.
"""
from __future__ import annotations

from typing import Any, Optional

from hermes_state_ids import new_session_id
from hermes_state_mutation_branch import _branch_boundary
from hermes_state_runtime import RuntimeStoreError
from hermes_state_titles import lineage_title_on_conn

# model_config carries two kinds of keys: the conversation's own model settings (max_tokens,
# max_iterations, reasoning_config, model/provider/...) which the fork must keep, and bookkeeping
# that belongs to one physical session (``_usage_anchor``, ``_reset_from``, ``_delegate_from``) or
# to one live runtime (``gateway_runtime``: peer routing, editor MCP scope). Copying the latter
# would hand the child another session's accounting anchor or a borrowed runtime binding.
_FORK_MODEL_CONFIG_EXCLUDE = ("gateway_runtime",)


def _fork_model_config(parent_model_config: Any, session_id: str) -> dict[str, Any]:
    from hermes_state_sessions import _parse_model_config

    carried = {
        key: value
        for key, value in (_parse_model_config(parent_model_config) or {}).items()
        if not str(key).startswith("_") and key not in _FORK_MODEL_CONFIG_EXCLUDE
    }
    carried["_branched_from"] = session_id
    return carried


class SessionForkMixin:
    def fork_session(
        self, session_id: str, *, through_message_id: Optional[int] = None,
        model: Optional[str] = None, title: Optional[str] = None,
    ) -> dict[str, Any]:
        """Copy the parent's active transcript into a NEW session; the parent is never written.

        *through_message_id* is the last row to copy (default: the active head; a boundary on an
        assistant row also takes the tool results that answer it). Returns the child's identity and
        the number of copied rows.
        """
        child = new_session_id()

        def _do(conn):
            parent = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if parent is None:
                raise ValueError(f"unknown session: {session_id}")
            try:
                boundary = _branch_boundary(conn, session_id, through_message_id)
            except RuntimeStoreError as exc:
                raise ValueError(
                    f"message {through_message_id} is not an active row of {session_id}") from exc
            self._publish_child_session_row(
                conn, parent, parent_session_id=session_id, child_session_id=child,
                source=parent["source"], model=model or parent["model"],
                model_config=_fork_model_config(parent["model_config"], session_id),
                system_prompt=None,  # NULL hash: the first turn rebuilds the prompt for this model
                cwd=parent["cwd"], profile_name=parent["profile_name"])
            ids, tools = self._tail_rows_after_watermark(
                conn,
                "SELECT id, tool_calls FROM messages "
                "WHERE session_id = ? AND active = 1 AND id <= ? ORDER BY id",
                [session_id, boundary])
            self._clone_message_rows(conn, ids, session_id=child)
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (len(ids), tools, child))
            self._set_session_title_in_transaction(
                conn, child, title or lineage_title_on_conn(conn, parent["title"] or "fork"),
                source=self.TITLE_SOURCE_USER if title else self.TITLE_SOURCE_DERIVED)
            return len(ids)

        copied = self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
        return {"session_id": child, "parent_session_id": session_id, "copied_messages": copied}
