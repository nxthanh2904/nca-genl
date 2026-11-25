param(
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dataPath = Join-Path $repoRoot 'assets/js/docs-data.js'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$source = [System.IO.File]::ReadAllText($dataPath, $utf8)
$pattern = '(?s)\A\s*// Embedded Full Markdown Documents Payload\s+const DOCS_DATA = (\[.*\]);\s*\z'
$match = [regex]::Match($source, $pattern)
if (-not $match.Success) {
    throw 'Cannot locate DOCS_DATA in assets/js/docs-data.js.'
}

$docs = ConvertFrom-Json $match.Groups[1].Value
$changed = @()
foreach ($doc in $docs) {
    $markdownPath = Join-Path $repoRoot $doc.file
    $content = [System.IO.File]::ReadAllText($markdownPath, $utf8)
    if ($doc.content -cne $content) {
        $changed += $doc.file
        $doc.content = $content
    }
}

if ($changed.Count -eq 0) {
    Write-Output 'DOCS_DATA matches all Markdown sources.'
    exit 0
}

if ($Check) {
    throw ('DOCS_DATA is out of sync: ' + ($changed -join ', '))
}

# Keep document metadata and order; only refresh content from Markdown.
$json = ConvertTo-Json -InputObject @($docs) -Depth 20 -Compress
$output = "        // Embedded Full Markdown Documents Payload`r`n        const DOCS_DATA = $json;`r`n"
[System.IO.File]::WriteAllText($dataPath, $output, $utf8)
Write-Output ('Updated DOCS_DATA: ' + ($changed -join ', '))
