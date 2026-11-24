# setup.ps1 — one-time environment setup for the qc-video skill (Windows).
#
# 1. Checks the REQUIRED system binaries (ffmpeg, ffprobe).
# 2. Creates a Python venv at the skill root and installs the OPTIONAL
#    enhanced-detector packages (numpy, opencv-python).
#
# The core scanner works with ffmpeg alone; the venv only adds the blur and
# flash-frame detectors. Safe to re-run.
#
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$SkillDir = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $SkillDir ".venv"

Write-Host "== qc-video setup =="
Write-Host "skill dir: $SkillDir"

# --- required system binaries ------------------------------------------------
$missing = $false
foreach ($bin in @("ffmpeg", "ffprobe")) {
    $cmd = Get-Command $bin -ErrorAction SilentlyContinue
    if ($cmd) {
        Write-Host "  [ok]  $bin -> $($cmd.Source)"
    } else {
        Write-Host "  [MISSING] $bin  - install with: winget install Gyan.FFmpeg"
        $missing = $true
    }
}

# --- optional: QCTools CLI ---------------------------------------------------
$qcli = Get-Command qcli -ErrorAction SilentlyContinue
if ($qcli) {
    Write-Host "  [ok]  qcli (optional) -> $($qcli.Source)"
} else {
    Write-Host "  [--]  qcli not found (optional; only needed for the QCTools GUI route)"
}

if ($missing) {
    Write-Error "Install the missing REQUIRED binaries above, then re-run."
    exit 1
}

# --- pick a Python launcher --------------------------------------------------
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Error "Python not found. Install Python 3 from https://www.python.org/ (check 'Add to PATH')."
    exit 1
}

# --- python venv + optional packages -----------------------------------------
if (-not (Test-Path $Venv)) {
    Write-Host "creating venv at $Venv"
    & $py.Source -m venv $Venv
}
$VenvPy = Join-Path $Venv "Scripts\python.exe"
Write-Host "installing optional detector packages (numpy, opencv-python)..."
& $VenvPy -m pip install --quiet --upgrade pip
& $VenvPy -m pip install --quiet -r (Join-Path $SkillDir "requirements.txt")

Write-Host ""
& $VenvPy -c "import numpy, cv2; print('  numpy  ' + numpy.__version__); print('  opencv ' + cv2.__version__)"

Write-Host ""
Write-Host "Done. Run scans with the venv interpreter so the enhanced detectors are active:"
Write-Host "  $VenvPy $(Join-Path $SkillDir 'scripts\qc-scan.py') VIDEO [VIDEO ...]"
