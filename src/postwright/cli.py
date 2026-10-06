import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from postwright.graph import create_graph, get_checkpointer
from postwright.publisher import PublishingRefusedError, publish_due_posts, publish_post
from postwright.scheduler import check_staleness
from postwright.state import HumanReviewDecision, InputNote, PostwrightState
from postwright.store import ApprovalHistoryStore, QueueStore

app = typer.Typer(name="postwright", help="Turn build notes into social media posts.")
queue_app = typer.Typer(name="queue", help="Manage scheduled post queue.")
app.add_typer(queue_app, name="queue")

console = Console()


def get_pending_threads(checkpointer: Any, graph: Any) -> list[dict[str, Any]]:
    """Return all threads currently interrupted at human review."""
    pending = []
    seen = set()
    cp_list = list(checkpointer.list(config=None))
    for cp in cp_list:
        tid = cp.config.get("configurable", {}).get("thread_id")
        if tid and tid not in seen:
            seen.add(tid)
            st = graph.get_state({"configurable": {"thread_id": tid}})
            if st.next and any(t.interrupts for t in st.tasks):
                interrupt_val = st.tasks[0].interrupts[0].value
                pending.append(
                    {
                        "thread_id": tid,
                        "interrupt_value": interrupt_val,
                        "state": st,
                    }
                )
    return pending


def display_staleness_warnings() -> None:
    """Check schedule priors and display warnings if stale."""
    warnings = check_staleness()
    for w in warnings:
        console.print(f"[bold yellow]⚠️  WARNING:[/] {w}")


@app.command()
def run(
    note: Annotated[
        str | None, typer.Option("--note", "-n", help="Raw build note string.")
    ] = None,
    file: Annotated[
        Path | None, typer.Option("--file", "-f", help="Path to markdown or text note file.")
    ] = None,
    project_tag: Annotated[
        str | None, typer.Option("--project-tag", "-p", help="Optional project tag.")
    ] = None,
    thread_id: Annotated[
        str | None, typer.Option("--thread-id", "-t", help="Thread ID for execution.")
    ] = None,
) -> None:
    """Process build notes and generate social media drafts."""
    display_staleness_warnings()

    if not note and not file:
        console.print("[bold red]Error:[/] Either --note or --file must be provided.")
        raise typer.Exit(code=1)

    note_content = note or ""
    source = "cli"

    if file:
        if not file.exists():
            console.print(f"[bold red]Error:[/] File '{file}' does not exist.")
            raise typer.Exit(code=1)
        note_content = file.read_text(encoding="utf-8")
        source = str(file)

    eff_thread_id = thread_id or f"run-{uuid.uuid4().hex[:8]}"
    input_note = InputNote(content=note_content, source=source, project_tag=project_tag)
    initial_state = PostwrightState(thread_id=eff_thread_id, raw_note=input_note)

    checkpointer = get_checkpointer()
    graph = create_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": eff_thread_id}}

    console.print(f"[bold blue]Running postwright graph (Thread ID: {eff_thread_id})...[/]")
    result = graph.invoke(initial_state, config=config)

    state = graph.get_state(config)
    if state.next and any(t.interrupts for t in state.tasks):
        console.print("\n[bold yellow]Run paused at human review.[/]")
        console.print(f"Thread ID: [bold cyan]{eff_thread_id}[/]")
        console.print(
            f"Run '[bold green]postwright review --thread-id {eff_thread_id}[/]' to review and resume."
        )
        return

    drafts = result.get("candidate_drafts", []) if isinstance(result, dict) else []
    if drafts:
        console.print(
            f"\n[bold green]Run completed successfully. Drafts generated: {len(drafts)}[/]"
        )


@app.command()
def review(
    thread_id: Annotated[
        str | None, typer.Option("--thread-id", "-t", help="Thread ID to review.")
    ] = None,
) -> None:
    """Review interrupted runs and provide human decision."""
    display_staleness_warnings()

    checkpointer = get_checkpointer()
    graph = create_graph(checkpointer=checkpointer)

    target_thread_id = thread_id

    if not target_thread_id:
        pending = get_pending_threads(checkpointer, graph)
        if not pending:
            console.print("[bold yellow]No pending interrupted runs found.[/]")
            return

        if len(pending) == 1:
            target_thread_id = pending[0]["thread_id"]
        else:
            table = Table(title="Pending Interrupted Runs")
            table.add_column("Index", style="cyan")
            table.add_column("Thread ID", style="bold green")
            table.add_column("Drafts Count", style="magenta")

            for idx, item in enumerate(pending, start=1):
                drafts_item = item["interrupt_value"].get("drafts", [])
                table.add_row(str(idx), str(item["thread_id"]), str(len(drafts_item)))

            console.print(table)
            choices = [str(i) for i in range(1, len(pending) + 1)]
            selected_idx = Prompt.ask("Select run to review", choices=choices, default="1")
            target_thread_id = pending[int(selected_idx) - 1]["thread_id"]

    config = {"configurable": {"thread_id": target_thread_id}}
    state = graph.get_state(config)

    if not state.next or not any(t.interrupts for t in state.tasks):
        console.print(
            f"[bold red]No pending human review interrupt found for thread '{target_thread_id}'.[/]"
        )
        return

    interrupt_val = state.tasks[0].interrupts[0].value
    drafts = interrupt_val.get("drafts", [])

    if not drafts:
        console.print("[bold yellow]No candidate drafts found in this run.[/]")
        return

    console.print(
        f"\n[bold green]Interrupted Run Drafts for Thread '{target_thread_id}':[/]\n"
    )

    for i, d in enumerate(drafts, start=1):
        d_id = d.get("id")
        platform = str(d.get("platform", "x")).upper()
        angle_format = d.get("angle_format", "general")
        content = d.get("content", "")
        score = d.get("score")
        critique = d.get("critique")
        rev_count = d.get("revision_count", 0)
        below_thresh = d.get("below_threshold", False)
        slot_data = d.get("proposed_slot")
        thread_parts = d.get("thread_parts", [])
        validation_errors = d.get("validation_errors", [])

        score_str = f"{score}/20" if score is not None else "N/A"
        warning = " [bold red]⚠️ BELOW THRESHOLD[/]" if below_thresh else ""

        slot_str = "None"
        if slot_data and isinstance(slot_data, dict):
            local_dt = slot_data.get("scheduled_at_local")
            tz = slot_data.get("user_timezone", "Africa/Nairobi")
            slot_str = f"{local_dt} ({tz})"

        body = content
        if thread_parts and len(thread_parts) > 1:
            body += "\n\n[bold magenta]Thread Parts Split:[/]"
            for idx, tp in enumerate(thread_parts, start=1):
                body += f"\n  [{idx}] {tp}"

        if validation_errors:
            body += "\n\n[bold red]Validation Warnings:[/"
            for ve in validation_errors:
                body += f"\n  • {ve}"

        body += f"\n\n[bold cyan]Proposed Slot:[/] {slot_str}"
        if critique:
            body += f"\n\n[bold yellow]Critique:[/] {critique}"

        title = f"[{i}] ID: {d_id} | Platform: {platform} | Angle: {angle_format} | Score: {score_str} | Revisions: {rev_count}{warning}"
        panel = Panel(
            body, title=title, border_style="red" if below_thresh else "cyan", expand=False
        )
        console.print(panel)

    action = Prompt.ask(
        "\nDecision action",
        choices=["approve", "edit", "reject", "rewrite"],
        default="approve",
    )

    draft_ids = [str(d.get("id")) for d in drafts if d.get("id")]
    selected_draft_id = None
    if len(drafts) == 1:
        selected_draft_id = draft_ids[0] if draft_ids else None
    elif draft_ids:
        selected_draft_id = Prompt.ask(
            "Select draft ID", choices=draft_ids, default=draft_ids[0]
        )

    slot_override = None
    if action in ("approve", "edit"):
        change_slot = Prompt.ask(
            "Change proposed slot?", choices=["y", "n"], default="n"
        )
        if change_slot.lower() == "y":
            slot_input = Prompt.ask(
                "Enter new slot (ISO format, e.g. 2025-05-10T14:00:00+03:00)"
            )
            slot_override = slot_input.strip()

    decision: HumanReviewDecision | None = None
    if action == "approve":
        decision = HumanReviewDecision(
            decision="approve", draft_id=selected_draft_id, slot_override=slot_override
        )

    elif action == "edit":
        current_draft = next(
            (d for d in drafts if d.get("id") == selected_draft_id), drafts[0]
        )
        current_content = current_draft.get("content", "")
        console.print(f"\n[bold cyan]Current Content:[/]\n{current_content}\n")
        new_text = Prompt.ask("Enter replacement text", default=current_content)
        decision = HumanReviewDecision(
            decision="edit",
            draft_id=selected_draft_id,
            edit_text=new_text,
            slot_override=slot_override,
        )

    elif action == "reject":
        decision = HumanReviewDecision(decision="reject", draft_id=selected_draft_id)

    elif action == "rewrite":
        note = Prompt.ask("Enter note / instructions for rewrite")
        decision = HumanReviewDecision(
            decision="rewrite", draft_id=selected_draft_id, rewrite_note=note
        )

    if decision is None:
        console.print("[bold red]No decision was selected.[/]")
        return

    console.print(
        f"[bold blue]Resuming graph execution for thread '{target_thread_id}'...[/]"
    )
    try:
        res = graph.invoke(Command(resume=decision.model_dump()), config=config)
    except ValueError as val_err:
        console.print(f"[bold red]Decision refused:[/] {val_err}")
        return

    new_state = graph.get_state(config)
    if new_state.next and any(t.interrupts for t in new_state.tasks):
        console.print(f"\n[bold yellow]Draft routed back to review after {action}.[/]")
        console.print(
            f"Run '[bold green]postwright review --thread-id {target_thread_id}[/]' to continue review."
        )
    else:
        status = res.get("status") if isinstance(res, dict) else "completed"
        console.print(
            f"\n[bold green]Decision '{action}' applied successfully. Run status: {status}[/]"
        )


@queue_app.callback(invoke_without_command=True)
def queue_list(ctx: typer.Context) -> None:
    """List posts in queue with status and platform post ID."""
    if ctx.invoked_subcommand is not None:
        return

    store = QueueStore()
    all_posts = store.get_queued_posts(status=None)

    if not all_posts:
        console.print("[bold yellow]No posts found in queue.[/]")
        return

    now_utc = datetime.now(UTC)
    table = Table(title="Scheduled Post Queue")
    table.add_column("ID", style="cyan")
    table.add_column("Platform", style="magenta")
    table.add_column("Status", style="bold yellow")
    table.add_column("Local Time", style="bold green")
    table.add_column("Countdown", style="yellow")
    table.add_column("Platform Post ID", style="blue")
    table.add_column("Draft ID", style="dim")

    for rec in all_posts:
        dt_utc = datetime.fromisoformat(rec.slot_utc)
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=UTC)
        else:
            dt_utc = dt_utc.astimezone(UTC)

        diff = dt_utc - now_utc
        if diff.total_seconds() <= 0:
            countdown = "Due now / past"
        else:
            hours, remainder = divmod(int(diff.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            days, hours = divmod(hours, 24)
            if days > 0:
                countdown = f"in {days}d {hours}h {minutes}m"
            elif hours > 0:
                countdown = f"in {hours}h {minutes}m"
            else:
                countdown = f"in {minutes}m {seconds}s"

        status_style = "green" if rec.status == "published" else ("red" if rec.status == "failed" else "yellow")
        status_str = f"[{status_style}]{rec.status}[/]"

        table.add_row(
            str(rec.id),
            rec.platform.upper(),
            status_str,
            f"{rec.slot_local} ({rec.user_timezone})",
            countdown if rec.status == "queued" else "-",
            rec.platform_post_id or "-",
            rec.draft_id,
        )

    console.print(table)


@queue_app.command(name="cancel")
def queue_cancel(
    post_id: Annotated[int, typer.Argument(help="ID of the queued post to cancel.")]
) -> None:
    """Cancel a queued post by ID."""
    store = QueueStore()
    success = store.cancel_post(post_id)
    if success:
        console.print(f"[bold green]Successfully cancelled queued post ID {post_id}.[/]")
    else:
        console.print(
            f"[bold red]Failed to cancel post ID {post_id}. Post not found or not in queued status.[/]"
        )


@app.command(name="publish")
def publish_one(
    post_id: Annotated[int, typer.Argument(help="ID of the queued post to publish now.")]
) -> None:
    """Publish a single queued post immediately (follows safety rules)."""
    store = QueueStore()
    post = store.get_post(post_id)
    if not post:
        console.print(f"[bold red]Error:[/] Queued post ID {post_id} not found.")
        raise typer.Exit(code=1)

    try:
        res = publish_post(post, store=store)
        if res.success:
            mode_str = "[yellow](DRY RUN)[/]" if res.dry_run else "[bold green](LIVE)[/]"
            console.print(
                f"[bold green]Post {post_id} published successfully {mode_str}. Platform Post ID: {res.platform_post_id}[/]"
            )
        else:
            console.print(
                f"[bold red]Failed to publish post {post_id}:[/] {res.error_message}"
            )
            raise typer.Exit(code=1)
    except PublishingRefusedError as e:
        console.print(f"[bold red]Publishing Refused:[/] {e}")
        raise typer.Exit(code=1)


@app.command(name="publish-due")
def publish_due() -> None:
    """Publish all queued posts whose slot time has passed."""
    store = QueueStore()
    results = publish_due_posts(store=store)

    if not results:
        console.print("[bold yellow]No due queued posts found for publishing.[/]")
        return

    console.print(f"\n[bold green]Processed {len(results)} due post(s):[/]\n")
    for r in results:
        mode_str = "[yellow](DRY RUN)[/]" if r.dry_run else "[bold green](LIVE)[/]"
        if r.success:
            console.print(
                f" • Success {mode_str} | Platform Post ID: {r.platform_post_id}"
            )
        else:
            console.print(f" • [bold red]Failed:[/] {r.error_message}")


@app.command()
def history(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Limit number of history items.")
    ] = 20,
) -> None:
    """List approved posts history with diffs."""
    store = ApprovalHistoryStore()
    records = store.get_history(limit=limit)

    if not records:
        console.print("[bold yellow]No approved post history found.[/]")
        return

    console.print(f"\n[bold green]Approved Post History ({len(records)} record(s)):[/]\n")

    for i, rec in enumerate(records, start=1):
        score_str = f"{rec.score}/20" if rec.score is not None else "N/A"
        title = f"History #{i} | Thread: {rec.thread_id} | Draft ID: {rec.draft_id} | Score: {score_str} | Date: {rec.created_at}"

        body = f"[bold green]Final Approved Text:[/]\n{rec.final_text}"
        if rec.unified_diff:
            body += f"\n\n[bold yellow]Unified Diff:[/]\n{rec.unified_diff}"

        panel = Panel(body, title=title, border_style="green", expand=False)
        console.print(panel)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    pass


if __name__ == "__main__":
    app()
