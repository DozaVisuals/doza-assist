"""DaVinci Resolve scripting-API integration.

Resolve has no Finder-open auto-import for EDL or XML — launching the app
with a timeline file does nothing. The only way to match FCP's "click
export → timeline appears" feel is to drive Resolve's scripting API
directly. The API ships with every Resolve install (Free and Studio,
v18+), uses local IPC (Mach ports on macOS), and needs no network.

Public surface:

  - ``probe()`` → ``ProbeResult`` describing whether the API is usable on
    this machine, and what the next step is if not.

  - ``import_timeline(timeline_path, source_media_path, project_name,
    timeline_name)`` → ``ImportResult``. Connects to a running Resolve
    (launches it if needed), imports the source media into the current
    project's Media Pool, imports the timeline file as a new timeline,
    and switches to it.

Reasons import_timeline can fail (each surfaces a distinct code so the
frontend can show the right setup hint):

  - ``module_missing``   — DaVinciResolveScript.py not found at the
                           expected install path. Either Resolve isn't
                           installed, or the user is on a very old
                           version (pre-18).
  - ``scripting_disabled`` — module loaded but ``scriptapp("Resolve")``
                            returned None. The user needs to enable
                            "External scripting using: Local" in
                            Resolve Preferences → System → General.
  - ``no_project``       — connected but no project is open and
                           CreateProject() failed.
  - ``import_failed``    — ImportTimelineFromFile returned None;
                           usually a source-media reconnect issue.
  - ``requires_studio`` — Resolve Free is installed (verified via the
                          app's CFBundleName) and the scripting daemon
                          isn't listening. As of Resolve 20, Blackmagic
                          gates the External Scripting toggle to Studio,
                          so on Free the auto-import path can't work no
                          matter what the user does — the frontend
                          should show drag-into-Media-Pool guidance
                          instead of a "flip this preference" walk-through.
  - ``unexpected_error`` — exception during the flow.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


# Resolve's scripting IPC calls have NO built-in timeout: a Resolve that
# hangs reconnecting media or chewing on a large XML would block the
# caller's thread forever (the Send-to-Resolve "beachball"). Run such a
# call on a DAEMON worker and give up after a hard cap so the export
# request always returns. A daemon thread is used deliberately — on
# timeout it is abandoned (the call keeps running inside Resolve) and
# dies with the process, so it can never block the request OR app exit
# (unlike a ThreadPoolExecutor, whose shutdown joins the hung worker).
_RESOLVE_CALL_TIMEOUT = 120.0

# Overall hard cap for the ENTIRE Resolve import handshake (project lookup +
# media add + timeline import + switch), run on one daemon worker so a
# wizard-blocked or wedged Resolve can never stall the request thread
# indefinitely. Generous enough to cover a legitimate large-timeline import
# plus a media reconnect, but bounded.
_IMPORT_TIMEOUT = 180.0

# scriptapp() handle-acquisition poll budgets. A COLD launch needs Resolve to
# boot its scripting daemon (several seconds). When Resolve is ALREADY
# running, a script-ready daemon answers in well under a second — so if it
# doesn't answer fast it's blocked (first-run setup wizard) or scripting is
# off; bail quickly with an actionable message instead of waiting the full
# cold budget (the old 30s "beachball").
_COLD_HANDLE_TIMEOUT = 30.0
_RUNNING_HANDLE_TIMEOUT = 8.0


def _call_with_timeout(fn, *args, timeout=_RESOLVE_CALL_TIMEOUT, **kwargs):
    """Run a blocking Resolve scripting call with a hard wall-clock cap.

    Returns the call's value, re-raises whatever it raised, or raises
    ``TimeoutError`` if it doesn't return within ``timeout`` seconds."""
    box = {}

    def _runner():
        try:
            box['value'] = fn(*args, **kwargs)
        except BaseException as exc:  # propagate to the caller below
            box['error'] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError('Resolve scripting call timed out')
    if 'error' in box:
        raise box['error']
    return box.get('value')


# Resolve's bundled scripting module. Paths are stable across recent
# Resolve releases — if Blackmagic ever moves them, we'd add a probe
# here. The ``.so`` next to the .py is the actual native bridge; the
# .py just dlopen's it.
_MODULES_DIR = (
    '/Library/Application Support/Blackmagic Design/DaVinci Resolve/'
    'Developer/Scripting/Modules'
)
_FUSION_SO = (
    '/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/'
    'Fusion/fusionscript.so'
)
_RESOLVE_APP = '/Applications/DaVinci Resolve/DaVinci Resolve.app'
_RESOLVE_PLIST = f'{_RESOLVE_APP}/Contents/Info.plist'


def edition(app_path: str | None = None) -> str:
    """Return ``'studio'``, ``'free'``, or ``'unknown'`` for a Resolve install.

    Studio installs ship as ``DaVinci Resolve Studio.app`` with
    ``CFBundleName = "DaVinci Resolve Studio"``; Free is just
    ``"DaVinci Resolve"``. Reading the plist is cheap and runs out of
    process, so we don't have to import anything from Resolve to find
    out — useful when scripting itself is unreachable.

    ``app_path`` is the specific Resolve .app to inspect (the running
    instance, or the version-ranked install for a cold launch). Falls back
    to the canonical ``_RESOLVE_APP`` path only when not given, so the
    edition we report always matches the Resolve we actually drive.
    """
    base = app_path or _RESOLVE_APP
    plist = os.path.join(base, 'Contents', 'Info.plist')
    if not os.path.isfile(plist):
        return 'unknown'
    try:
        out = subprocess.check_output(
            ['defaults', 'read', plist, 'CFBundleName'],
            text=True, timeout=2, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return 'unknown'
    return 'studio' if 'studio' in out.lower() else 'free'


@dataclass
class ProbeResult:
    ok: bool
    reason: str | None = None       # None when ok=True
    hint: str | None = None         # user-facing next-step message


@dataclass
class ImportResult:
    ok: bool
    reason: str | None = None
    hint: str | None = None
    timeline_name: str | None = None


def _ensure_modules_on_path() -> bool:
    """Add Resolve's scripting module dir to sys.path if not already there.

    Returns False if the dir doesn't exist (Resolve not installed, or
    install layout changed).
    """
    if not os.path.isdir(_MODULES_DIR):
        return False
    if _MODULES_DIR not in sys.path:
        sys.path.insert(0, _MODULES_DIR)
    return True


def _try_import_module():
    """Lazy import of DaVinciResolveScript. Returns the module or None."""
    if not _ensure_modules_on_path():
        return None
    try:
        import DaVinciResolveScript as dvr_script  # type: ignore[import-not-found]
        return dvr_script
    except Exception:
        return None


def _is_resolve_running() -> bool:
    """Best-effort check whether Resolve.app is currently running."""
    try:
        out = subprocess.check_output(
            ['pgrep', '-i', '-f', 'DaVinci Resolve.app/Contents/MacOS'],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return bool(out.strip())
    except Exception:
        return False


def running_app_path() -> str | None:
    """Return the .app path of the Resolve instance currently running, or None.

    Connect-before-launch relies on this: if a Resolve is already open we
    drive THAT instance (the user has their project in it) and never pick
    or launch a different version. Derives the bundle path from the running
    process's executable path, so it reports the exact copy in use —
    whatever version, wherever installed.
    """
    try:
        out = subprocess.check_output(
            ['pgrep', '-i', '-f', 'DaVinci Resolve.app/Contents/MacOS'],
            stderr=subprocess.DEVNULL, timeout=2, text=True,
        ).strip()
    except Exception:
        return None
    for pid in (p for p in out.splitlines() if p.strip()):
        try:
            comm = subprocess.check_output(
                ['ps', '-p', pid.strip(), '-o', 'comm='],
                stderr=subprocess.DEVNULL, timeout=2, text=True,
            ).strip()
        except Exception:
            continue
        idx = comm.find('.app/')
        if idx != -1:
            return comm[:idx + 4]  # up to and including '.app'
    return None


def _launch_resolve(app_path: str | None = None) -> None:
    """Bring a specific Resolve to the front (launches if not running).

    Non-blocking. ``app_path`` is the version-ranked install chosen for a
    COLD launch; only used when no Resolve is running (connect-before-launch
    handles the already-running case upstream). Falls back to the canonical
    path when not given.
    """
    try:
        subprocess.Popen(['open', '-a', app_path or _RESOLVE_APP])
    except Exception:
        pass


def probe() -> ProbeResult:
    """Inspect whether the scripting API is usable right now.

    Cheap — no Resolve launch, no project mutation. Returns ok=True only
    if a Resolve scriptapp handle is reachable. Anything else gets a
    reason code plus a user-facing hint.
    """
    if not os.path.isfile(_FUSION_SO):
        return ProbeResult(
            ok=False, reason='module_missing',
            hint=('DaVinci Resolve 18 or newer is required for auto-import. '
                  'Update Resolve from blackmagicdesign.com and try again.'),
        )
    dvr_script = _try_import_module()
    if dvr_script is None:
        return ProbeResult(
            ok=False, reason='module_missing',
            hint=('Resolve scripting modules not found. Reinstall DaVinci '
                  'Resolve to restore them.'),
        )
    # If Resolve isn't running, we can't tell whether scripting is enabled —
    # so don't fail the probe here. import_timeline() will launch Resolve
    # and retry once.
    if not _is_resolve_running():
        return ProbeResult(ok=True)
    try:
        handle = dvr_script.scriptapp('Resolve')
    except Exception as e:
        return ProbeResult(
            ok=False, reason='unexpected_error',
            hint=f'Resolve scripting raised: {e}',
        )
    if handle is None:
        if edition() == 'free':
            return ProbeResult(
                ok=False, reason='requires_studio',
                hint=('Auto-import into DaVinci Resolve requires Resolve '
                      'Studio — Blackmagic gates the scripting API behind '
                      'the paid tier as of Resolve 20. In Free, import the '
                      'export manually: Resolve → File → Import → '
                      'Timeline… → choose the XML in Finder. Resolve will '
                      'create the timeline and link the source media '
                      'automatically.'),
            )
        return ProbeResult(
            ok=False, reason='scripting_disabled',
            hint=('Resolve needs External Scripting turned on for auto-import. '
                  'In Resolve: Preferences (⌘,) → System → General → '
                  '"External scripting using:" → choose Local → Save → '
                  'restart Resolve, then try the export again.'),
        )
    return ProbeResult(ok=True)


def _get_resolve_handle(dvr_script, timeout_seconds: float = 30.0):
    """Poll scriptapp("Resolve") until it returns a handle, up to timeout.

    Resolve takes several seconds after launch before its scripting
    daemon is listening; calling scriptapp immediately after `open -a`
    returns None even on a properly-configured install.
    """
    deadline = time.monotonic() + timeout_seconds
    last_attempt = None
    while time.monotonic() < deadline:
        try:
            handle = dvr_script.scriptapp('Resolve')
        except Exception as e:
            last_attempt = e
            handle = None
        if handle is not None:
            return handle, None
        time.sleep(0.5)
    if last_attempt is not None:
        return None, f'scriptapp raised: {last_attempt}'
    return None, None


def _do_import(handle, timeline_path, source_media_path,
               project_name, timeline_name) -> ImportResult:
    """The full Resolve import handshake — run on ONE bounded worker.

    Every scripting call lives here: project lookup/create, the
    ``GetMediaStorage()`` / ``GetMediaPool()`` receiver evaluations, the
    media add, the timeline import, and the switch. Running the whole
    sequence under a single wall-clock cap (see import_timeline) means a
    wizard-blocked or wedged Resolve can stall this worker but never the
    request thread. Returns an ImportResult; unexpected exceptions
    propagate to import_timeline's handler.
    """
    pm = handle.GetProjectManager()
    project = pm.GetCurrentProject()
    if project is None:
        project = pm.CreateProject(project_name)
    if project is None:
        return ImportResult(
            ok=False, reason='no_project',
            hint=('Resolve has no open project and a new one could not '
                  'be created. Open or create a project in Resolve, '
                  'then run the export again.'),
        )

    # Add source media into the Media Pool. Best-effort: if the source
    # path isn't reachable (network drive offline, path moved since
    # transcription) we still try the timeline import — Resolve marks
    # clips offline but the cuts and structure land.
    if source_media_path and os.path.isfile(source_media_path):
        try:
            handle.GetMediaStorage().AddItemListToMediaPool([source_media_path])
        except Exception:
            pass

    media_pool = project.GetMediaPool()
    import_opts = {'timelineName': timeline_name}
    if source_media_path:
        import_opts['sourceClipsPath'] = os.path.dirname(source_media_path)

    timeline = media_pool.ImportTimelineFromFile(timeline_path, import_opts)
    if timeline is None:
        return ImportResult(
            ok=False, reason='import_failed',
            hint=('Resolve refused the timeline file. The file is in '
                  'Finder — try File → Import → Timeline manually to '
                  'see the underlying error.'),
        )

    try:
        project.SetCurrentTimeline(timeline)
    except Exception:
        # Non-fatal: the timeline is imported, we just couldn't switch to
        # it. The user can pick it from the Media Pool.
        pass

    return ImportResult(ok=True, timeline_name=timeline_name)


def import_timeline(
    timeline_path: str,
    *,
    source_media_path: str | None,
    project_name: str,
    timeline_name: str,
    app_path: str | None = None,
) -> ImportResult:
    """Connect to Resolve, import source media + timeline, switch to it.

    ``app_path`` is the specific Resolve .app to drive: the already-running
    instance (connect-before-launch, resolved upstream) or the
    version-ranked install chosen for a cold launch. It governs the cold
    launch and the edition check, so we always talk to — and report on —
    the same copy. Falls back to the canonical install only when omitted.

    Flow:
      1. Load DaVinciResolveScript (fail fast if absent).
      2. Cold-launch ``app_path`` only if no Resolve is running; wait for
         the scripting daemon — a long budget when cold (Resolve is
         booting), a SHORT budget when one is already running so a
         wizard-blocked daemon bails fast instead of beachballing.
      3-6. Project lookup/create, media add, timeline import, switch — all
         on one bounded daemon worker (``_do_import``) so a wedged Resolve
         can never stall the request thread indefinitely.

    Returns ImportResult with ok=True on success. On failure the reason
    code feeds the frontend's setup-modal trigger.
    """
    dvr_script = _try_import_module()
    if dvr_script is None:
        return ImportResult(
            ok=False, reason='module_missing',
            hint=('DaVinci Resolve 18 or newer is required for auto-import. '
                  'Falling back to "reveal in Finder" — drop the file into '
                  'the Media Pool to import.'),
        )

    # Connect-before-launch: only cold-launch when nothing is running, and
    # poll on the short budget when a Resolve is already up.
    running = _is_resolve_running()
    if not running:
        _launch_resolve(app_path)

    handle, err = _get_resolve_handle(
        dvr_script,
        timeout_seconds=(_RUNNING_HANDLE_TIMEOUT if running else _COLD_HANDLE_TIMEOUT),
    )
    if handle is None:
        # Launched Resolve (or it was running) but scriptapp never returned
        # a handle. On Free this is structural (Blackmagic restricts
        # scripting to Studio); on Studio it's External Scripting being off
        # — or Resolve still booting / stuck on its first-run setup wizard.
        if edition(app_path) == 'free':
            return ImportResult(
                ok=False, reason='requires_studio',
                hint=('Auto-import into DaVinci Resolve requires Resolve '
                      'Studio. In Free, import manually: File → Import → '
                      'Timeline… → choose the XML file in Finder. Resolve '
                      'creates the timeline and auto-links the source '
                      'media from the XML\'s embedded paths.'),
            )
        return ImportResult(
            ok=False, reason='scripting_disabled',
            hint=('Resolve is open but did not accept scripting in time. If '
                  'a first-run setup screen is showing, finish it. Then '
                  'enable External Scripting: Resolve → Preferences (⌘,) → '
                  'System → General → "External scripting using:" → Local → '
                  'Save → restart Resolve, then run the export again.'),
        )

    # Run the ENTIRE handshake on one bounded worker (not just the import
    # calls) so the receiver evals and a wizard-blocked project lookup can
    # never stall past _IMPORT_TIMEOUT.
    try:
        return _call_with_timeout(
            _do_import, handle, timeline_path, source_media_path,
            project_name, timeline_name, timeout=_IMPORT_TIMEOUT,
        )
    except TimeoutError:
        return ImportResult(
            ok=False, reason='import_timeout',
            hint=('Resolve did not finish importing in time (it may be '
                  'reconnecting media, busy, or showing a setup dialog). '
                  'The timeline file is in Finder — import it manually with '
                  'File → Import → Timeline once Resolve is responsive.'),
        )
    except Exception as e:
        return ImportResult(
            ok=False, reason='unexpected_error',
            hint=f'Resolve auto-import failed: {e}',
        )
