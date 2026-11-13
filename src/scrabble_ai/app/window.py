"""The app's own window: a native Windows window that can show the desktop through its glass.

A browser window cannot be transparent to the desktop, so the pages are shown in a WebView2
control (the same engine as Microsoft Edge, which ships with Windows 11) inside a window of our
own, made with pywebview. Three layers have to let the desktop through:

    the window     Asking the compositor to blur behind an *empty* region of the window switches
                   the window to per-pixel alpha: every pixel the window itself paints black
                   with GDI has alpha 0, so the desktop shows there. (winit and tao make
                   transparent windows on Windows the same way.)
    the WebView    Its default background is set to transparent, so where the page paints
                   nothing the window behind it shows, which is to say the desktop.
    the page       `?glass=1` on the address tells the page it may leave its background clear
                   (css/liquid.css, <html data-backdrop="frost" or "clear">).

For frosted glass the desktop behind the window is blurred by Windows itself: the window is
given the acrylic "accent" (`SetWindowCompositionAttribute`), which blurs and lightly tints
whatever is behind every transparent pixel. That is the recipe Tauri apps use. With the accent
off the desktop shows sharp.

The title bar and window frame are the ordinary solid ones. If any of this cannot be set up the
page is told (see Bridge) and draws its own backdrop instead, and if the window cannot be made
at all the caller falls back to an Edge app window.
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

WCA_ACCENT_POLICY = 19
ACCENT_DISABLED = 0
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
# The tint Windows lays over the blurred desktop, as (red, green, blue, alpha). The page adds its
# own wash on top, so these stay light. Acrylic misbehaves with an alpha of 0.
FROST_TINT = {"light": (255, 255, 255, 56), "dark": (14, 16, 20, 96)}


class GlassUnavailable(RuntimeError):
    """The glass window cannot be made on this machine (the reason is the message)."""


class _BlurBehind(ctypes.Structure):
    _fields_ = [("dwFlags", ctypes.c_ulong), ("fEnable", ctypes.c_int), ("hRgnBlur", ctypes.c_void_p),
                ("fTransitionOnMaximized", ctypes.c_int)]


class _AccentPolicy(ctypes.Structure):
    _fields_ = [("AccentState", ctypes.c_int), ("AccentFlags", ctypes.c_int), ("GradientColor", ctypes.c_uint),
                ("AnimationId", ctypes.c_int)]


class _CompositionAttribute(ctypes.Structure):
    _fields_ = [("Attribute", ctypes.c_int), ("Data", ctypes.c_void_p), ("SizeOfData", ctypes.c_size_t)]


def glass_url(url: str) -> str:
    """`url` with `glass=1` in its query, which tells the page it is in this window."""
    base, mark, fragment = url.partition("#")
    if "glass=1" not in base.partition("?")[2].split("&"):
        base += ("&" if "?" in base else "?") + "glass=1"
    return base + mark + fragment


def enable_see_through(hwnd: int, dwmapi=None, gdi32=None) -> bool:
    """Give the window per-pixel transparency (see the module docstring). True on success."""
    dwmapi = dwmapi or ctypes.windll.dwmapi
    gdi32 = gdi32 or ctypes.windll.gdi32
    gdi32.CreateRectRgn.restype = ctypes.c_void_p
    region = gdi32.CreateRectRgn(0, 0, -1, -1)  # an empty region: nothing is blurred by this call
    try:
        blur = _BlurBehind(DWM_BB_ENABLE | DWM_BB_BLURREGION, 1, region, 0)
        return dwmapi.DwmEnableBlurBehindWindow(ctypes.c_void_p(hwnd), ctypes.byref(blur)) == 0
    finally:
        gdi32.DeleteObject(ctypes.c_void_p(region))


def set_frost(hwnd: int, on: bool, theme: str = "light", user32=None) -> bool:
    """Turn the blur of whatever is behind the window on or off. True if Windows accepted it."""
    user32 = user32 or ctypes.windll.user32
    red, green, blue, alpha = FROST_TINT["dark" if theme == "dark" else "light"]
    policy = _AccentPolicy(ACCENT_ENABLE_ACRYLICBLURBEHIND if on else ACCENT_DISABLED, 0,
                           (max(1, alpha) << 24) | (blue << 16) | (green << 8) | red, 0)
    data = _CompositionAttribute(WCA_ACCENT_POLICY, ctypes.cast(ctypes.pointer(policy), ctypes.c_void_p),
                                 ctypes.sizeof(policy))
    function = user32.SetWindowCompositionAttribute  # undocumented, and in every Windows since 10
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CompositionAttribute)]
    function.restype = ctypes.c_int
    return bool(function(ctypes.c_void_p(hwnd), ctypes.byref(data)))


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
        self._frost_available = True  # until Windows refuses it
        self._frost = False
        self._theme = "light"
        self._problem = ""

    def _state(self) -> dict:
        return {"ok": self._see_through, "frost": self._frost, "frost_available": self._frost_available,
                "problem": self._problem}

    def glass_state(self) -> dict:
        """Whether the window is see-through, and whether the desktop behind it is frosted."""
        return self._state()

    def set_backdrop(self, mode: str, theme: str = "light") -> dict:
        """`frost` blurs the desktop behind the window, anything else leaves it sharp (the page
        covers it itself for its built-in backdrop). Returns the state, for the page to act on."""
        self._theme = "dark" if theme == "dark" else "light"
        if not self._see_through or not self._hwnd:
            return self._state()
        want = mode == "frost"
        try:
            accepted = set_frost(self._hwnd, want, self._theme)
        except Exception as error:
            accepted = False
            self._problem = f"{type(error).__name__}: {error}"
        if want and not accepted:
            self._frost_available = False
            self._problem = self._problem or "Windows refused the acrylic effect"
        self._frost = want and accepted
        return self._state()

    def set_theme(self, theme: str) -> bool:
        self._theme = "dark" if theme == "dark" else "light"
        if not self._hwnd:
            return False
        try:
            if self._frost:
                set_frost(self._hwnd, True, self._theme)  # the tint under the glass follows the theme
            return set_dark_title_bar(self._hwnd, self._theme == "dark")
        except Exception:  # cosmetic: never worth an error in the page
            return False


def prepare(form, bridge: Bridge) -> None:
    """Make a freshly created window see-through and frosted, before it is first shown. `form`
    is the WinForms window pywebview made; this runs on its own thread."""
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
        # pywebview asks for the Mica backdrop on a dark system; none is wanted behind the glass.
        try:
            none = ctypes.c_int(DWMSBT_NONE)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), DWMWA_SYSTEMBACKDROP_TYPE,
                                                       ctypes.byref(none), ctypes.sizeof(none))
        except Exception:
            pass
        if not enable_see_through(hwnd):
            raise OSError("the desktop compositor refused per-pixel transparency")
        set_dark_title_bar(hwnd, False)  # the app is light until the page says otherwise
        bridge._see_through = True
    except Exception as error:
        bridge._see_through = False
        bridge._problem = f"{type(error).__name__}: {error}"
        print(f"glass window: staying opaque ({bridge._problem})", flush=True)
        return
    # Frosted from the first frame; the page switches it off if Settings says clear or built-in.
    bridge.set_backdrop("frost", bridge._theme)
    if not bridge._frost:
        print(f"glass window: no frost ({bridge._problem})", flush=True)


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
    # Not private: the page keeps its theme and glass choices in local storage.
    webview.start(private_mode=False, storage_path=str(storage_dir),
                  icon=str(icon) if icon and Path(icon).is_file() else None)
    return shown["yes"]
