from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser


@dataclass
class Option:
    value: str
    text: str
    selected: bool = False


@dataclass
class Select:
    name: str
    multiple: bool
    options: list[Option] = field(default_factory=list)


@dataclass
class Input:
    name: str
    value: str
    kind: str
    checked: bool
    nearby_text: str = ""


@dataclass
class Form:
    action: str
    method: str
    inputs: list[Input] = field(default_factory=list)
    selects: list[Select] = field(default_factory=list)


class FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms: list[Form] = []
        self.links: list[tuple[str, str]] = []
        self._form: Form | None = None
        self._select: Select | None = None
        self._option: Option | None = None
        self._anchor_href: str | None = None
        self._anchor_text: list[str] = []
        self._recent_input: Input | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:
        data = {
            str(key).lower(): str(value or "")
            for key, value in attrs
        }
        tag = tag.lower()
        if tag == "form":
            self._form = Form(
                action=data.get("action", ""),
                method=data.get(
                    "method",
                    "get",
                ).lower(),
            )
            self.forms.append(self._form)
        elif (
            tag == "select"
            and self._form is not None
        ):
            self._select = Select(
                name=data.get("name", ""),
                multiple="multiple" in data,
            )
            self._form.selects.append(
                self._select
            )
        elif (
            tag == "option"
            and self._select is not None
        ):
            self._option = Option(
                value=data.get("value", ""),
                text="",
                selected="selected" in data,
            )
            self._select.options.append(
                self._option
            )
        elif (
            tag == "input"
            and self._form is not None
        ):
            item = Input(
                name=data.get("name", ""),
                value=data.get("value", ""),
                kind=data.get(
                    "type",
                    "text",
                ).lower(),
                checked="checked" in data,
            )
            self._form.inputs.append(item)
            self._recent_input = item
        elif tag == "a":
            self._anchor_href = data.get(
                "href",
                "",
            )
            self._anchor_text = []

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        tag = tag.lower()
        if tag == "form":
            self._form = None
            self._recent_input = None
        elif tag == "select":
            self._select = None
        elif tag == "option":
            self._option = None
        elif (
            tag == "a"
            and self._anchor_href is not None
        ):
            self.links.append(
                (
                    self._anchor_href,
                    " ".join(
                        self._anchor_text
                    ).strip(),
                )
            )
            self._anchor_href = None
            self._anchor_text = []

    def handle_data(
        self,
        data: str,
    ) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._option is not None:
            self._option.text += (
                (" " if self._option.text else "")
                + text
            )
        if self._anchor_href is not None:
            self._anchor_text.append(text)
        if self._recent_input is not None:
            joined = (
                self._recent_input.nearby_text
                + " "
                + text
            ).strip()
            self._recent_input.nearby_text = (
                joined[-240:]
            )


def parse(
    html: str,
) -> FormParser:
    parser = FormParser()
    parser.feed(html)
    return parser
