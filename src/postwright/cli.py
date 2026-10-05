from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from postwright.graph import create_graph
from postwright.state import InputNote, PostwrightState

app = typer.Typer(name="postwright", help="Turn build notes into social media posts.")
console = Console()


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
) -> None:
    """Process build notes and generate social media drafts."""
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

    input_note = InputNote(content=note_content, source=source, project_tag=project_tag)
    initial_state = PostwrightState(raw_note=input_note)

    console.print("[bold blue]Running postwright graph...[/]")
    graph = create_graph()
    final_state = graph.invoke(initial_state)

    drafts = final_state.get("candidate_drafts", [])

    if not drafts:
        console.print("[bold yellow]No drafts were generated.[/]")
        return

    console.print(f"\n[bold green]Generated {len(drafts)} candidate draft(s):[/]\n")

    for i, draft in enumerate(drafts, start=1):
        platform = str(getattr(draft, "platform", "x")).upper()
        angle_format = getattr(draft, "angle_format", "general")
        content = getattr(draft, "content", "")
        is_thread = getattr(draft, "is_thread", False)
        thread_parts = getattr(draft, "thread_parts", [])

        body = content
        if is_thread and thread_parts:
            body += "\n\n[bold cyan]Thread Parts:[/]\n" + "\n---\n".join(
                f"{idx+1}. {part}" for idx, part in enumerate(thread_parts)
            )

        title = f"Draft #{i} | Platform: {platform} | Angle: {angle_format}"
        panel = Panel(body, title=title, border_style="cyan", expand=False)
        console.print(panel)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    pass


if __name__ == "__main__":
    app()
