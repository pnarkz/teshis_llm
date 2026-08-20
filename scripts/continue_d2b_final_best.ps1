$project = 'C:\Users\ASUS\Desktop\termal_teshis'
$python = 'C:\Program Files\Python311\python.exe'
$firstPid = 28368

while (Get-Process -Id $firstPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$args = @(
    '-u', 'scripts/local_d2b.py',
    '--dataset', 'C:\Users\ASUS\Desktop\HYZ\dataset',
    '--model', "$project\final_best.pt",
    '--output-dataset', "$project\veri_surumleri\v03_d2b_eksik_etiket_final_best",
    '--output-root', "$project\experiments",
    '--run-name', 'run_D2b_42_final_best_local',
    '--epochs', '30', '--batch', '8', '--imgsz', '768', '--seed', '42'
)

Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $project `
    -RedirectStandardOutput "$project\experiments\d2b_final_best_training.log" `
    -RedirectStandardError "$project\experiments\d2b_final_best_training.err.log" `
    -WindowStyle Hidden
