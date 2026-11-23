# Start the packaged app, drive it through its HTTP API on a generated clip.
$ErrorActionPreference = "Stop"
$env:FFMPEG = "$PWD\bin\ffmpeg.exe"
python tests/make_clip.py "$PWD\smoke.mp4"
$p = Start-Process -FilePath "dist\Retitle\Retitle.exe" -ArgumentList "--server" -PassThru
Start-Sleep 15
$B = "http://127.0.0.1:8765"
Invoke-RestMethod "$B/api/fonts" | Out-Null
$v = ("$PWD\smoke.mp4" -replace '\\','/')
Invoke-RestMethod -Method Post "$B/api/open" -Body (@{video=$v} | ConvertTo-Json) -ContentType "application/json"
Invoke-RestMethod -Method Post "$B/api/run" -Body '{"job":"analyze"}' -ContentType "application/json"
for ($i=0; $i -lt 60; $i++) { Start-Sleep 5; $s = Invoke-RestMethod "$B/api/status"; if (-not $s.running) { break } }
$s.log | ForEach-Object { Write-Host $_ }
if ($s.error) { throw "analyze failed: $($s.error)" }
$pr = Invoke-RestMethod "$B/api/project"
$n = $pr.titles.titles.Count
Write-Host "titles: $n"
if ($n -ne 3) { throw "expected 3 titles" }
Stop-Process -Id $p.Id -Force
Write-Host "EXE SMOKE OK"
