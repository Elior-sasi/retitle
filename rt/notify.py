"""Windows notifications: a toast when a long job ends while Retitle is not in front,
plus a flashing taskbar button. No extra packages - the toast goes through
Windows PowerShell's WinRT bridge. The installer gives the Start-menu shortcut
the AppUserModelID below, so the toast shows "Retitle" with its icon."""
import base64
import os
import subprocess
import sys
import threading
import time

AUMID = "Elior.Retitle"
PS_AUMID = "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe"
MIN_SECONDS = 20          # quick jobs end while the user is still looking
FOCUS = {"t": 0.0, "focused": True}   # reported by the page with every status poll


def set_app_id():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(AUMID)
    except Exception:
        pass


def report_focus(focused):
    FOCUS.update(t=time.time(), focused=bool(focused))


def _away():
    # no poll for a while means the page is hidden/minimised, which also counts as away
    return not FOCUS["focused"] or time.time() - FOCUS["t"] > 5


def _xml(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _script(title, body):
    xml = (f'<toast lang="he-IL" duration="long"><visual><binding template="ToastGeneric">'
           f'<text>{_xml(title)}</text><text>{_xml(body)}</text></binding></visual>'
           f'<audio src="ms-winsoundevent:Notification.Default"/></toast>')
    return (
        "$ErrorActionPreference='Stop';"
        "[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime]>$null;"
        "[Windows.Data.Xml.Dom.XmlDocument,Windows.Data.Xml.Dom.XmlDocument,ContentType=WindowsRuntime]>$null;"
        f"$x=New-Object Windows.Data.Xml.Dom.XmlDocument;$x.LoadXml(@'\n{xml}\n'@);"
        "$t=New-Object Windows.UI.Notifications.ToastNotification $x;"
        f"try{{[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{AUMID}').Show($t)}}"
        f"catch{{[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{PS_AUMID}').Show($t)}}")


def toast(title, body):
    if sys.platform != "win32":
        return
    enc = base64.b64encode(_script(title, body).encode("utf-16-le")).decode()
    try:
        subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                          "-EncodedCommand", enc], creationflags=0x08000000,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def flash():
    """Flash the Retitle taskbar button until the window comes to the front."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        pid = os.getpid()
        found = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(h, _):
            p = wintypes.DWORD()
            u.GetWindowThreadProcessId(h, ctypes.byref(p))
            if p.value == pid and u.IsWindowVisible(h):
                found.append(h)
            return True
        u.EnumWindows(cb, 0)

        class FW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("hwnd", wintypes.HWND), ("dwFlags", wintypes.DWORD),
                        ("uCount", wintypes.UINT), ("dwTimeout", wintypes.DWORD)]
        for h in found:
            f = FW(ctypes.sizeof(FW), h, 0x3 | 0xC, 0, 0)    # caption+tray, until foreground
            u.FlashWindowEx(ctypes.byref(f))
    except Exception:
        pass


def job_ended(name_he, seconds, error=None, cancelled=False, enabled=True, extra=""):
    """Called when any job ends. Silent for cancels, quick jobs and when the user is looking."""
    if not enabled or cancelled or seconds < MIN_SECONDS or not _away():
        return False
    mins = int(round(seconds / 60))
    took = f"לקח {mins} דק'" if mins >= 1 else f"לקח {int(seconds)} שניות"
    if error:
        title, body = f"{name_he} נעצר בשגיאה", str(error)[:180]
    else:
        title, body = f"{name_he} הסתיים", (extra[:150] + " - " if extra else "") + took
    threading.Thread(target=lambda: (toast(title, body), flash()), daemon=True).start()
    return True


def ask(title, body):
    """The run is waiting for the user: toast + flash, but only when they are not looking."""
    if _away():
        threading.Thread(target=lambda: (toast(title, body), flash()), daemon=True).start()
