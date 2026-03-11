Add-Type -AssemblyName System.Windows.Forms
$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = [System.Drawing.SystemIcons]::Information
$notify.Visible = $true
$notify.ShowBalloonTip(5000, "Claude Code", "Task completed", [System.Windows.Forms.ToolTipIcon]::Info)
[System.Media.SystemSounds]::Beep.Play()
Start-Sleep -Seconds 6
$notify.Dispose()
