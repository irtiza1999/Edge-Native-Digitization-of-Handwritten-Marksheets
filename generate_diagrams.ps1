# Extract Mermaid diagrams and convert to PNG

$markdownFile = "c:\Users\USERAS\Desktop\handwritten\pipeline_comparison_diagram.md"
$outputDir = "c:\Users\USERAS\Desktop\handwritten\diagrams"

# Create output directory
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# Read the markdown file
$content = Get-Content $markdownFile -Raw

# Extract all mermaid code blocks
$mermaidBlocks = @()
$blockPattern = '```mermaid\n([\s\S]*?)\n```'
$matches = [regex]::Matches($content, $blockPattern)

foreach ($i in 0..($matches.Count - 1)) {
    $mermaidCode = $matches[$i].Groups[1].Value
    $blockNum = $i + 1
    
    # Determine block name from context
    $names = @(
        "1_Main_Comparison_Flow",
        "2_Hybrid_Pipeline_Architecture",
        "3_Base_YOLOv8_Architecture"
    )
    
    $blockName = if ($i -lt $names.Count) { $names[$i] } else { "diagram_$blockNum" }
    $mermaidFile = "$outputDir\$blockName.mmd"
    $pngFile = "$outputDir\$blockName.png"
    
    # Write mermaid file
    Set-Content -Path $mermaidFile -Value $mermaidCode -Encoding UTF8
    
    Write-Host "Converting diagram $blockNum : $blockName"
    Write-Host "  Mermaid: $mermaidFile"
    Write-Host "  PNG: $pngFile"
    
    # Convert to PNG
    mmdc -i $mermaidFile -o $pngFile -s 2
}

Write-Host "`nAll diagrams converted successfully!"
Write-Host "PNG files saved to: $outputDir"
Get-ChildItem $outputDir -Filter "*.png" | ForEach-Object { Write-Host "  [OK] $($_.Name)" }
