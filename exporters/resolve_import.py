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
import time
from dataclasses import dataclass


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


def edition() -> str:
    """Return ``'studio'``, ``'free'``, or ``'unknown'``.

    Studio installs ship as ``DaVinci Resolve Studio.app`` with
    ``CFBundleName = "DaVinci Resolve Studio"``; Free is just
    ``"DaVinci Resolve"``. Reading the plist is cheap and runs out of
    process, so we don't have to import anything from Resolve to find
    out — useful when scripting itself is unreachable.
    """
    if not os.path.isfile(_RESOLVE_PLIST):
        return 'unknown'
    try:
        out = subprocess.check_output(
            ['defaults', 'read', _RESOLVE_PLIST, 'CFBundleName'],
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


def _launch_resolve() -> None:
    """Bring Resolve to the front (launches if not running). Non-blocking."""
    try:
        subprocess.Popen(['open', '-a', _RESOLVE_APP])
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


def import_timeline(
    timeline_path: str,
    *,
    source_media_path: str | None,
    project_name: str,
    timeline_name: str,
) -> ImportResult:
    """Connect to Resolve, import source media + timeline, switch to it.

    Flow:
      1. Load DaVinciResolveScript (fail fast if absent).
      2. Launch Resolve if not already running, wait for the scripting
         daemon to accept connections (up to 30s).
      3. Use the current project, or create one named after the Doza
         Assist project if Resolve has none open.
      4. Add the source media file(s) to the Media Pool (if path given
         and exists). Skips silently if media isn't reachable — the
         timeline still imports, clips just land offline like a manual
         EDL import would.
      5. ``MediaPool.ImportTimelineFromFile(timeline_path, {
            timelineName: ..., sourceClipsPath: media_dir
          })``
      6. Set the imported timeline as current so the user lands on it.

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

    if not _is_resolve_running():
        _launch_resolve()

    handle, err = _get_resolve_handle(dvr_script)
    if handle is None:
        # We launched Resolve (or it was running) but scriptapp never
        # returned a handle. On Free, this is structural (no toggle
        # to flip — Blackmagic restricts scripting to Studio). On
        # Studio, it's the External Scripting preference being off.
        if edition() == 'free':
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
            hint=('Resolve is open but rejected the import. Enable External '
                  'Scripting: Resolve → Preferences (⌘,) → System → General → '
                  '"External scripting using:" → Local → Save → restart '
                  'Resolve, then run the export again.'),
        )

    try:
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

        # Add source media into the Media Pool. This is best-effort:
        # if the source path isn't reachable (network drive offline,
        # path moved since transcription) we still try the timeline
        # import — Resolve will mark clips offline but at least the
        # cuts and structure land.
        if source_media_path and os.path.isfile(source_media_path):
            try:
                handle.GetMediaStorage().AddItemListToMediaPool(
                    [source_media_path]
                )
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
            # Non-fatal: the timeline is imported, we just couldn't
            # switch to it. The user can pick it from the Media Pool.
            pass

        return ImportResult(ok=True, timeline_name=timeline_name)

    except Exception as e:
        return ImportResult(
            ok=False, reason='unexpected_error',
            hint=f'Resolve auto-import failed: {e}',
        )
