"""The glass window (scrabble_ai.app.window) and the launcher's choice of window.

Nothing here opens a window: pywebview and the Windows compositor calls are replaced by fakes.
Whether the window really is see-through on a screen can only be judged by looking at it.
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import types
from pathlib import Path

import pytest

from scrabble_ai.app import __main__ as launcher
from scrabble_ai.app import window as glass


def test_glass_url_adds_the_flag_before_the_route():
    assert glass.glass_url("http://127.0.0.1:5000") == "http://127.0.0.1:5000?glass=1"
    assert glass.glass_url("http://127.0.0.1:5000/#/play?opponent=bot:greedy") == "http://127.0.0.1:5000/?glass=1#/play?opponent=bot:greedy"
    assert glass.glass_url("http://h/?mock=1#/run/a") == "http://h/?mock=1&glass=1#/run/a"
    assert glass.glass_url("http://h/?glass=1#/a") == "http://h/?glass=1#/a", "not added twice"


class FakeDwm:
    def __init__(self, result=0):
        self.result = result
        self.blur_calls = []
        self.attribute_calls = []

    def DwmEnableBlurBehindWindow(self, hwnd, blur):  # noqa: N802 - the Windows name
        structure = ctypes.cast(blur, ctypes.POINTER(glass._BlurBehind)).contents
        self.blur_calls.append((hwnd.value, structure.dwFlags, structure.fEnable, structure.hRgnBlur))
        return self.result

    def DwmSetWindowAttribute(self, hwnd, attribute, value, size):  # noqa: N802
        self.attribute_calls.append((hwnd.value, attribute, ctypes.cast(value, ctypes.POINTER(ctypes.c_int)).contents.value, size))
        return self.result


class FakeGdi:
    def __init__(self):
        self.created, self.deleted = [], []

        def create(*corners):
            self.created.append(corners)
            return 0xBEEF

        self.CreateRectRgn = create
        self.DeleteObject = lambda handle: self.deleted.append(handle.value)


def test_see_through_asks_for_blur_behind_an_empty_region():
    dwm, gdi = FakeDwm(), FakeGdi()
    assert glass.enable_see_through(4242, dwm, gdi) is True
    assert gdi.created == [(0, 0, -1, -1)], "an empty region: per-pixel alpha with nothing blurred"
    assert dwm.blur_calls == [(4242, glass.DWM_BB_ENABLE | glass.DWM_BB_BLURREGION, 1, 0xBEEF)]
    assert gdi.deleted == [0xBEEF], "the region is released"
    assert glass.enable_see_through(4242, FakeDwm(result=-2147024891), FakeGdi()) is False


@pytest.mark.parametrize("dark, expected", [(True, 1), (False, 0)])
def test_title_bar_follows_the_theme(dark, expected):
    dwm = FakeDwm()
    assert glass.set_dark_title_bar(77, dark, dwm) is True
    assert dwm.attribute_calls == [(77, glass.DWMWA_USE_IMMERSIVE_DARK_MODE, expected, 4)]


class FakeUser32:
    def __init__(self, result=1):
        self.calls = []

        def set_attribute(hwnd, data):
            attribute = ctypes.cast(data, ctypes.POINTER(glass._CompositionAttribute)).contents
            policy = ctypes.cast(attribute.Data, ctypes.POINTER(glass._AccentPolicy)).contents
            self.calls.append((hwnd.value, attribute.Attribute, attribute.SizeOfData, policy.AccentState,
                               policy.AccentFlags, policy.GradientColor))
            return result

        self.SetWindowCompositionAttribute = set_attribute


def test_frost_is_the_acrylic_accent_with_a_light_tint():
    user32 = FakeUser32()
    assert glass.set_frost(55, True, "light", user32) is True
    red, green, blue, alpha = glass.FROST_TINT["light"]
    assert user32.calls == [(55, glass.WCA_ACCENT_POLICY, ctypes.sizeof(glass._AccentPolicy),
                             glass.ACCENT_ENABLE_ACRYLICBLURBEHIND, 0, (alpha << 24) | (blue << 16) | (green << 8) | red)]
    assert alpha >= 1, "acrylic misbehaves with no tint at all"
    assert glass.set_frost(55, True, "dark", user32) is True and user32.calls[-1][5] != user32.calls[0][5]
    assert glass.set_frost(55, False, "light", user32) is True and user32.calls[-1][3] == glass.ACCENT_DISABLED
    assert glass.set_frost(55, True, "light", FakeUser32(result=0)) is False, "Windows can refuse"


def test_bridge_reports_and_never_raises(monkeypatch):
    bridge = glass.Bridge()
    assert bridge.glass_state() == {"ok": False, "frost": False, "frost_available": True, "problem": ""}
    assert bridge.set_theme("dark") is False, "no window yet"
    assert bridge.set_backdrop("frost")["ok"] is False, "nothing to frost before the window is see-through"
    calls = []
    monkeypatch.setattr(glass, "set_dark_title_bar", lambda hwnd, dark: calls.append((hwnd, dark)) or True)
    bridge._hwnd = 9
    assert bridge.set_theme("light") is True and bridge.set_theme("dark") is True
    assert calls == [(9, False), (9, True)]
    monkeypatch.setattr(glass, "set_dark_title_bar", lambda hwnd, dark: 1 / 0)
    assert bridge.set_theme("dark") is False
    # Only these are visible to the page.
    assert sorted(name for name in dir(bridge) if not name.startswith("_")) == ["glass_state", "set_backdrop", "set_theme"]


def test_bridge_backdrop_modes(monkeypatch):
    frosted = []
    monkeypatch.setattr(glass, "set_frost", lambda hwnd, on, theme: frosted.append((hwnd, on, theme)) or True)
    monkeypatch.setattr(glass, "set_dark_title_bar", lambda hwnd, dark: True)
    bridge = glass.Bridge()
    bridge._hwnd, bridge._see_through = 9, True
    assert bridge.set_backdrop("frost", "light") == {"ok": True, "frost": True, "frost_available": True, "problem": ""}
    assert bridge.set_backdrop("clear", "light")["frost"] is False
    assert bridge.set_backdrop("builtin", "dark")["frost"] is False, "the page covers the desktop itself"
    assert frosted == [(9, True, "light"), (9, False, "light"), (9, False, "dark")]
    # With frost on, a change of theme changes the tint under the glass as well as the title bar.
    bridge.set_backdrop("frost", "dark")
    bridge.set_theme("light")
    assert frosted[-2:] == [(9, True, "dark"), (9, True, "light")]

    # Windows refusing the effect is reported, once and for all, and clear glass still works.
    monkeypatch.setattr(glass, "set_frost", lambda hwnd, on, theme: not on)
    refused = glass.Bridge()
    refused._hwnd, refused._see_through = 9, True
    state = refused.set_backdrop("frost")
    assert state["ok"] is True and state["frost"] is False and state["frost_available"] is False and "refused" in state["problem"]
    assert refused.set_backdrop("clear")["frost_available"] is False

    def explode(hwnd, on, theme):
        raise OSError("no such function")

    monkeypatch.setattr(glass, "set_frost", explode)
    broken = glass.Bridge()
    broken._hwnd, broken._see_through = 9, True
    assert "OSError" in broken.set_backdrop("frost")["problem"]


class FakeHandle:
    def ToInt64(self):  # noqa: N802 - the .NET name
        return 31337


def fake_form():
    control = types.SimpleNamespace(DefaultBackgroundColor=None, BackColor=None)
    return types.SimpleNamespace(Handle=FakeHandle(), BackColor=None, browser=types.SimpleNamespace(webview=control))


@pytest.mark.skipif(os.name != "nt", reason="the glass window is Windows only")
def test_prepare_makes_every_layer_transparent(monkeypatch):
    pytest.importorskip("clr")
    seen = []
    monkeypatch.setattr(glass, "enable_see_through", lambda hwnd: seen.append(hwnd) or True)
    monkeypatch.setattr(glass, "set_dark_title_bar", lambda hwnd, dark: True)
    monkeypatch.setattr(glass, "set_frost", lambda hwnd, on, theme: True)
    form, bridge = fake_form(), glass.Bridge()
    glass.prepare(form, bridge)
    assert bridge.glass_state() == {"ok": True, "frost": True, "frost_available": True, "problem": ""}
    assert bridge._hwnd == 31337 and seen == [31337]
    # The window paints black (alpha 0 to the compositor) and the WebView paints nothing.
    assert (form.BackColor.R, form.BackColor.G, form.BackColor.B) == (0, 0, 0)
    assert form.browser.webview.DefaultBackgroundColor.A == 0


@pytest.mark.skipif(os.name != "nt", reason="the glass window is Windows only")
def test_prepare_that_fails_leaves_an_opaque_window_and_says_so(monkeypatch, capsys):
    pytest.importorskip("clr")
    monkeypatch.setattr(glass, "enable_see_through", lambda hwnd: False)
    monkeypatch.setattr(glass, "set_frost", lambda hwnd, on, theme: pytest.fail("no frost without transparency"))
    bridge = glass.Bridge()
    glass.prepare(fake_form(), bridge)  # must not raise: the window still opens
    state = bridge.glass_state()
    assert state["ok"] is False and state["frost"] is False and "transparency" in state["problem"]
    assert "staying opaque" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "nt", reason="the glass window is Windows only")
def test_prepare_without_frost_is_still_see_through(monkeypatch, capsys):
    pytest.importorskip("clr")
    monkeypatch.setattr(glass, "enable_see_through", lambda hwnd: True)
    monkeypatch.setattr(glass, "set_dark_title_bar", lambda hwnd, dark: True)
    monkeypatch.setattr(glass, "set_frost", lambda hwnd, on, theme: False)
    bridge = glass.Bridge()
    glass.prepare(fake_form(), bridge)
    state = bridge.glass_state()
    assert state["ok"] is True and state["frost"] is False and state["frost_available"] is False
    assert "no frost" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "nt", reason="the glass window is Windows only")
def test_this_machine_can_make_the_window():
    """Loads pywebview's Windows back end and finds the WebView2 runtime, without opening anything."""
    pytest.importorskip("webview")
    glass.check_available()


def test_check_available_explains_what_is_missing(monkeypatch):
    monkeypatch.setattr(glass.os, "name", "posix")
    with pytest.raises(glass.GlassUnavailable, match="only available on Windows"):
        glass.check_available()


class FakeEvent:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self


def fake_webview(calls, show=True):
    """A stand-in for the pywebview module that records how it is used."""
    module = types.ModuleType("webview")

    def create_window(title, url, **options):
        calls["create"] = (title, url, options)
        events = types.SimpleNamespace(before_show=FakeEvent(), shown=FakeEvent())
        calls["window"] = types.SimpleNamespace(events=events, native=fake_form())
        return calls["window"]

    def start(**options):
        calls["start"] = options
        window = calls["window"]
        for handler in window.events.before_show.handlers:
            handler(window)
        if show:
            for handler in window.events.shown.handlers:
                handler()

    module.create_window, module.start = create_window, start
    return module


def test_run_opens_the_page_in_glass_mode(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setitem(sys.modules, "webview", fake_webview(calls))
    monkeypatch.setattr(glass, "check_available", lambda: None)
    prepared = []
    monkeypatch.setattr(glass, "prepare", lambda form, bridge: prepared.append((form, bridge)))
    icon = tmp_path / "lab.ico"
    icon.write_bytes(b"ico")
    assert glass.run("http://127.0.0.1:5123/#/overview", tmp_path / "webview", icon) is True
    title, url, options = calls["create"]
    assert title == "Scrabble Lab" and url == "http://127.0.0.1:5123/?glass=1#/overview"
    assert options["background_color"] == "#000000", "black is what the compositor reads as clear"
    assert options["text_select"] is True and options["min_size"] == glass.MIN_SIZE
    assert isinstance(options["js_api"], glass.Bridge)
    assert prepared and prepared[0][0] is calls["window"].native and prepared[0][1] is options["js_api"]
    # Local storage is kept (theme, see-through amount), in the app's own folder.
    assert calls["start"] == {"private_mode": False, "storage_path": str(tmp_path / "webview"), "icon": str(icon)}
    assert (tmp_path / "webview").is_dir()

    calls.clear()
    monkeypatch.setitem(sys.modules, "webview", fake_webview(calls, show=False))
    assert glass.run("http://127.0.0.1:5123", tmp_path / "webview", tmp_path / "missing.ico") is False
    assert calls["start"]["icon"] is None, "a missing icon file is not passed on"


# ---- the launcher --------------------------------------------------------------------------


def test_default_window(monkeypatch):
    monkeypatch.delenv("SCRABBLE_LAB_WINDOW", raising=False)
    assert launcher.default_window() == ("glass" if os.name == "nt" else "edge")
    monkeypatch.setenv("SCRABBLE_LAB_WINDOW", "Edge")
    assert launcher.default_window() == "edge"
    monkeypatch.setenv("SCRABBLE_LAB_WINDOW", "nonsense")
    assert launcher.default_window() == ("glass" if os.name == "nt" else "edge")


class FakeProcess:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def test_show_uses_the_glass_window(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)
    monkeypatch.setattr(launcher, "make_icon", lambda path: Path(path).write_bytes(b"ico") or True)
    opened = []
    monkeypatch.setattr(glass, "run", lambda url, storage, icon: opened.append((url, storage, icon)) or True)
    monkeypatch.setattr(launcher, "open_window", lambda url: pytest.fail("the Edge window must not open too"))
    assert launcher.show("http://127.0.0.1:5000", "glass") is True
    assert opened == [("http://127.0.0.1:5000", tmp_path / "webview", tmp_path / "scrabble-lab.ico")]


def test_show_falls_back_to_edge_when_glass_cannot_be_made(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(launcher, "APP_DIR", tmp_path)
    monkeypatch.setattr(launcher, "make_icon", lambda path: False)

    def unavailable(url, storage, icon):
        raise glass.GlassUnavailable("the Microsoft Edge WebView2 runtime was not found")

    monkeypatch.setattr(glass, "run", unavailable)
    process = FakeProcess()
    monkeypatch.setattr(launcher, "open_window", lambda url: process)
    assert launcher.show("http://127.0.0.1:5000", "glass") is True and process.waited
    assert "using an Edge window instead" in capsys.readouterr().out

    # With no Edge either, the address goes to the default browser and there is nothing to wait for.
    monkeypatch.setattr(launcher, "open_window", lambda url: None)
    assert launcher.show("http://127.0.0.1:5000", "glass") is False


def test_show_edge_never_touches_the_glass_window(monkeypatch):
    monkeypatch.setattr(glass, "run", lambda *args: pytest.fail("glass was not asked for"))
    process = FakeProcess()
    monkeypatch.setattr(launcher, "open_window", lambda url: process)
    assert launcher.show("http://127.0.0.1:5000", "edge") is True and process.waited


def test_the_page_is_liquid_glass_and_light_by_default():
    static = Path(glass.__file__).resolve().parent / "static"
    page = (static / "index.html").read_text(encoding="utf-8")
    assert 'data-theme="light"' in page and 'data-backdrop="builtin"' in page
    assert page.index("css/liquid.css") > page.index("css/board.css"), "the finish is laid over the components"
    assert "js/views/settings.js" in page
    tokens = (static / "css" / "tokens.css").read_text(encoding="utf-8")
    light, dark = tokens.split(':root[data-theme="dark"]')
    assert "color-scheme: light" in light and "color-scheme: dark" in dark
    assert "--outline: #0b0b0b" in light, "black outlines"
    liquid = (static / "css" / "liquid.css").read_text(encoding="utf-8")
    # The desktop shows only when the page is told it can: otherwise the page paints its own backdrop.
    assert "body { background: var(--backdrop)" in liquid
    assert ':root[data-backdrop="frost"] body, :root[data-backdrop="clear"] body { background: var(--wash); }' in liquid
    assert "backdrop-filter: var(--glass-blur)" in liquid
    # No component stylesheet still draws a grey box outline: boxes are outlined in black.
    for name in ("app.css", "board.css"):
        css = (static / "css" / name).read_text(encoding="utf-8")
        assert not re.search(r"border: 1px solid var\(--border(-strong)?\)", css), name
