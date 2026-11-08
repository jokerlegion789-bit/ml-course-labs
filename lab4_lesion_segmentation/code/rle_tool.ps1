<#
.SYNOPSIS
    task4 提交文件的 RLE 解析、合法性校验与可控膨胀。

.DESCRIPTION
    提交格式为 Id,Predicted，Predicted 是「起始位置 游程长度」整数对，
    坐标系为 512x512 行优先、1-indexed（见官方 segmentation_to_csv.py）。

    本工具做三件事：
      1) 校验：起点 >= 1、游程 >= 1、起点严格递增且不重叠、不超 512*512
      2) 统计：每张图的前景像素数与占比
      3) 膨胀：每个游程两端各扩 N 像素，相邻/重叠自动合并

.PARAMETER Input
    输入 CSV 路径

.PARAMETER Output
    输出 CSV 路径

.PARAMETER Dilate
    每侧膨胀像素数，0 表示只校验不修改

.EXAMPLE
    .\rle_tool.ps1 -Input submission.csv -Output modified.csv -Dilate 1
#>
param(
    [Parameter(Mandatory = $true)][string]$Input,
    [string]$Output = "",
    [int]$Dilate = 0
)

$ErrorActionPreference = 'Stop'
$MAX = 512 * 512

$lines = Get-Content $Input
if ($lines[0].Trim() -ne 'Id,Predicted') {
    Write-Warning "表头不是 'Id,Predicted'，实际为: $($lines[0])"
}

$outLines = New-Object System.Collections.Generic.List[string]
$outLines.Add('Id,Predicted')

$nImg = 0; $bad = 0; $sumRatio = 0.0

for ($k = 1; $k -lt $lines.Count; $k++) {
    $line = $lines[$k]
    if ([string]::IsNullOrWhiteSpace($line)) { continue }

    $c = $line.IndexOf(',')
    $id = $line.Substring(0, $c)
    $parts = ($line.Substring($c + 1)).Trim() -split '\s+'

    $a = New-Object 'int[]' $parts.Count
    for ($j = 0; $j -lt $parts.Count; $j++) { $a[$j] = [int]$parts[$j] }

    if ($a.Length -eq 0) { $outLines.Add("$id,"); continue }

    # ---- 统计与校验 ----
    $tot = 0
    for ($j = 1; $j -lt $a.Length; $j += 2) { $tot += $a[$j] }

    $prevEnd = 0; $ok = $true
    for ($j = 0; $j -lt $a.Length; $j += 2) {
        $st = $a[$j]; $ln = $a[$j + 1]
        if ($st -lt 1 -or $ln -lt 1 -or $st -le $prevEnd -or ($st + $ln - 1) -gt $MAX) {
            $ok = $false; break
        }
        $prevEnd = $st + $ln - 1
    }
    if (-not $ok) { $bad++ }

    if ($Dilate -le 0) {
        $outLines.Add("$id," + ($parts -join ' '))
    }
    else {
        # ---- 膨胀：每侧扩 $Dilate 像素，重叠则合并 ----
        $sb = New-Object System.Text.StringBuilder
        $cs = 0; $ce = 0; $have = $false
        for ($j = 0; $j -lt $a.Length; $j += 2) {
            $st = [Math]::Max(1, $a[$j] - $Dilate)
            $en = [Math]::Min($MAX, $a[$j] + $a[$j + 1] + $Dilate - 1)
            if (-not $have) { $cs = $st; $ce = $en; $have = $true }
            elseif ($st -le $ce + 1) { if ($en -gt $ce) { $ce = $en } }
            else { [void]$sb.Append("$cs $($ce - $cs + 1) "); $cs = $st; $ce = $en }
        }
        if ($have) { [void]$sb.Append("$cs $($ce - $cs + 1)") }
        $outLines.Add("$id," + $sb.ToString().Trim())
    }

    $sumRatio += $tot / $MAX * 100
    $nImg++
}

$avg = if ($nImg -gt 0) { $sumRatio / $nImg } else { 0 }
Write-Host "图片数        : $nImg"
Write-Host "非法 RLE 行数 : $bad"
Write-Host ("平均前景占比  : {0:N2}%" -f $avg)

if ($Output) {
    $outLines | Set-Content -Path $Output -Encoding UTF8
    $kb = [math]::Round((Get-Item $Output).Length / 1KB, 1)
    Write-Host "已写出        : $Output  ($kb KB)"
}
