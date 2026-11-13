"""The app's own window: a native Windows window with a see-through glass background.

A browser window cannot be transparent to the desktop, so the pages are shown in a WebView2
control (the same engine as Microsoft Edge, which ships with Windows 11) inside a window of our
own, made with pywebview. Three layers have to let the desktop through:

    the window     Asking the compositor to blur behind an *empty* region of the window switches
                   the window to per-pixel alpha without blurring anything: every pixel the
                   window itself paints black with GDI has alpha 0, so the desktop shows there.
                   (winit and tao make transparent windows on Windows the same way.)
    the WebView    Its default background is set to transparent, so where the page paints
                   nothing the window behind it shows, which is to say the desktop.
    the page       `?glass=1` on the address turns on the glass styles (css/glass.css): panels
                   are translucent, the page between them nearly clear, and text, borders,
                   controls, the board and its tiles stay solid.

The title bar and window frame are the ordinary solid ones. If any of this cannot be set up the
window is simply opaque and the page is told so, and if the window cannot be made at all the
caller falls back to an Edge app window.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

TITLE = "Scrabble Lab"
APP_ID = "ScrabbleLab.App"  # groups the window under its own taskbar button, with its own icon
SIZE = (1500, 960)
MIN_SIZE = (1100, 720)

DWM_BB_ENABLE = 0x1
DWM_BB_BLURREGION = 0x2
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMSBT_NONE = 1


class GlassUnavailable(RuntimeError):
    """The glass window cannot be made on this machine (the reason is the message)."""


class _BlurBehind(ctypes.Structure):
    _fields_ = [("dwFlags", ctypes.c_ulong), ("fEnable", ctypes.c_int), ("hRgnBlur", ctypes.c_void_p),
                ("fTransitionOnMaximized", ctypes.c_int)]


def glass_url(url: str) -> str:
    """`url` with `glass=1` in its query, which is what turns the glass styles on in the page."""
    base, mark, fragment = url.partition("#")
    if "glass=1" not in base.partition("?")[2].split("&"):
        base += ("&" if "?" in base else "?") + "glass=1"
    return base + mark + fragment


def enable_see_through(hwnd: int, dwmapi=None, gdi32=None) -> bool:
    """Give the window per-pixel transparency (see the module docstring). True on success."""
    dwmapi = dwmapi or ctypes.windll.dwmapi
    gdi32 = gdi32 or ctypes.windll.gdi32
    gdi32.CreateRectRgn.restype = ctypes.c_void_p
    region = gdi32.CreateRectRgn(0, 0, -1, -1)  # an empty region: nothing is blurred
    try:
        blur = _BlurBehind(DWM_BB_ENABLE | DWM_BB_BLURREGION, 1, region, 0)
        return dwmapi.DwmEnableBlurBehindWindow(ctypes.c_void_p(hwnd), ctypes.byref(blur)) == 0
    finally:
        gdi32.DeleteObject(ctypes.c_void_p(region))


def set_dark_title_bar(hwnd: int, dark: bool, dwmapi=None) -> bool:
    """Match the system title bar to the app's theme. True on success."""
    dwmapi = dwmapi or ctypes.windll.dwmapi
    value = ctypes.c_int(1 if dark else 0)
    return dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), DWMWA_USE_IMMERSIVE_DARK_MODE,
                                        ctypes.byref(value), ctypes.sizeof(value)) == 0


class Bridge:
    """What the page can ask of its window, as `window.pywebview.api.<method>()`."""

    def __init__(self) -> None:
        self._hwnd = 0
        self._see_through = False
        self._problem = ""

    def glass_state(self) -> dict:
        """Whether the window really is see-through. The page drops its glass styles if not."""
        return {"ok": self._see_through, "problem": self._problem}

    def set_theme(self, theme: str) -> bool:
        if not self._hwnd:
            return False
        try:
            return set_dark_title_bar(self._hwnd, theme != "light")
        except Exception:  # cosmetic: never worth an error in the page
            return False


def prepare(form, bridge: Bridge) -> None:
    """Make a freshly created window see-through, before it is first shown. `form` is the
    WinForms window pywebview made; this runs on its own thread."""
    try:
        import clr  # pythonnet; pywebview has loaded it already

        clr.AddReference("System.Drawing")
        from System.Drawing import Color

        hwnd = int(form.Handle.ToInt64())
        bridge._hwnd = hwnd
        # Black painted by the window is what the compositor reads as fully transparent.
        form.BackColor = Color.Black
        control = form.browser.webview
        control.DefaultBackgroundColor = Color.Transparent
        try:
            control.BackColor = Color.Black
        except Exception:
            pass  # the control inherits the window's colour anyway
        # pywebview asks for the Mica backdrop on a dark system; none is wanted behind clear glass.
        try:
            none = ctypes.c_int(DWMSBT_NONE)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), DWMWA_SYSTEMBACKDROP_TYPE,
                                                       ctypes.byref(none), ctypes.sizeof(none))
        except Exception:
            pass
        if not enable_see_through(hwnd):
            raise OSError("the desktop compositor refused per-pixel transparency")
        set_dark_title_bar(hwnd, True)
        bridge._see_through = True
    except Exception as error:
        bridge._see_through = False
        bridge._problem = f"{type(error).__name__}: {error}"
        print(f"glass window: staying opaque ({bridge._problem})", flush=True)


def check_available() -> None:
    """Raise GlassUnavailable unless the glass window can be made here. Opens nothing."""
    if os.name != "nt":
        raise GlassUnavailable("the glass window is only available on Windows")
    try:
        import webview  # noqa: F401
        from webview.platforms import winforms
    except Exception as error:
        raise GlassUnavailable(f"pywebview could not be loaded ({type(error).__name__}: {error})") from None
    if getattr(winforms, "renderer", "") != "edgechromium":
        raise GlassUnavailable("the Microsoft Edge WebView2 runtime was not found")


def run(url: str, storage_dir: Path, icon: Path | None = None) -> bool:
    """Show `url` in the glass window and return when the window is closed. Returns whether a
    window was shown at all; raises GlassUnavailable if one cannot be made."""
    check_available()
    import webview

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass
    storage_dir.mkdir(parents=True, exist_ok=True)
    bridge = Bridge()
    shown = {"yes": False}
    window = webview.create_window(TITLE, glass_url(url), width=SIZE[0], height=SIZE[1], min_size=MIN_SIZE,
                                   background_color="#000000", js_api=bridge, text_select=True)

    def before_show(window):
        prepare(window.native, bridge)

    def on_shown():
        shown["yes"] = True

    window.events.before_show += before_show
    window.events.shown += on_shown
    # Not private: the page keeps its theme, glass and speed choices in local storage.
    webview.start(private_mode=False, storage_path=str(storage_dir),
                  icon=str(icon) if icon and Path(icon).is_file() else None)
    return shown["yes"]
