from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Button, Footer, Header, Input, RichLog
from textual import work

from extract_html.converter import convert_html_to_markdown
from extract_html.extractor import extract_from_url
from extract_html.file_writer import (
    ensure_output_directories,
    extract_name_from_url,
    get_default_output_dir,
    write_markdown_file,
)
from extract_html.image_downloader import download_images


@dataclass
class PipelineContext:
    url: str
    output_dir: Path


class ExtractApp(App):
    TITLE = "Extract HTML to Markdown"
    BINDINGS = [Binding("q", "quit", "Quit")]
    CSS = """
    Screen {
        padding: 1 2;
    }

    Input {
        margin-bottom: 1;
    }

    Button {
        margin-bottom: 1;
    }

    RichLog {
        min-height: 15;
        border: solid $primary;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Enter URL to extract...")
        yield Button("Extract", variant="primary")
        yield RichLog(highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._start_extraction(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        url = self.query_one(Input).value
        self._start_extraction(url)

    def _start_extraction(self, url: str) -> None:
        url = url.strip()
        if not url:
            return
        context = PipelineContext(url=url, output_dir=get_default_output_dir())
        self._run_pipeline(context)

    @work(thread=True)
    def _run_pipeline(self, context: PipelineContext) -> None:
        try:
            self._execute_pipeline(context)
        except Exception as exc:
            self._log(f"Error: {exc}")

    def _execute_pipeline(self, context: PipelineContext) -> None:
        self._log(f"Fetching URL: {context.url}...")
        extraction = extract_from_url(context.url)

        self._log("Extracting content...")
        name = extract_name_from_url(context.url)
        ensure_output_directories(context.output_dir, name)

        self._log("Converting to Markdown...")
        conversion = convert_html_to_markdown(
            extraction.html_content, extraction.image_urls, name
        )

        image_count = len(conversion.image_mappings)
        self._log(f"Downloading images ({image_count} found)...")
        download_result = download_images(
            conversion.image_mappings,
            context.output_dir,
            self._build_progress_callback(image_count),
        )

        output_path = write_markdown_file(context.output_dir, name, conversion.markdown)
        self._log(f"Saved: {output_path}")
        self._log(f"Done! Extracted {len(download_result.succeeded)} images.")

        self.call_from_thread(self._clear_input)

    def _build_progress_callback(self, total: int):
        def on_progress(filename: str, index: int, _total: int) -> None:
            self._log(f"  [{index}/{total}] Downloaded: {filename}")

        return on_progress

    def _log(self, message: str) -> None:
        self.call_from_thread(self.query_one(RichLog).write, message)

    def _clear_input(self) -> None:
        self.query_one(Input).value = ""


def main():
    ExtractApp().run()


if __name__ == "__main__":
    main()
