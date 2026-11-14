# Export submission/deck.pptx to deck.pdf (+ optional PNGs for review) with the installed PowerPoint.
#   powershell -ExecutionPolicy Bypass -File submission/export_deck.ps1 [-PngDir <folder>]
param([string]$PngDir = "")
$deck = (Resolve-Path "$PSScriptRoot\deck.pptx").Path
$pdf = Join-Path $PSScriptRoot "deck.pdf"
$pp = New-Object -ComObject PowerPoint.Application
try {
    $pres = $pp.Presentations.Open($deck, $true, $false, $false)   # read-only, no window
    $pres.SaveAs($pdf, 32)                                          # 32 = ppSaveAsPDF
    if ($PngDir) {
        New-Item -ItemType Directory -Force $PngDir | Out-Null
        foreach ($s in $pres.Slides) { $s.Export((Join-Path $PngDir ("slide-{0}.png" -f $s.SlideIndex)), "PNG", 1600, 900) }
    }
    $pres.Close()
    "exported $pdf"
} finally { $pp.Quit() }
