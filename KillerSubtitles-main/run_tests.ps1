# KillerSubtitles - Test Suite
# Generates several subtitle variations from Timeline.mp4

Write-Host "`n=== KillerSubtitles Test Suite ===" -ForegroundColor Cyan
Write-Host "Input: Timeline.mp4`n"

# Test 1: Karaoke, 4 words per line, larger font, white+gold
Write-Host "[Test 1] Karaoke - 4 words/line, large font (120px), gold highlight" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_01_karaoke_4wpl_large.mp4 `
    --mode karaoke `
    --words-per-line 4 `
    --max-lines 3 `
    --font-size 120 `
    --font-color "#FFFFFF" `
    --highlight-color "#FFD700" `
    --position lower
Write-Host ""

# Test 2: Karaoke, 4 words per line, even bigger, red highlight
Write-Host "[Test 2] Karaoke - 4 words/line, XL font (140px), red highlight" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_02_karaoke_4wpl_xl_red.mp4 `
    --mode karaoke `
    --words-per-line 4 `
    --max-lines 2 `
    --font-size 140 `
    --font-color "#FFFFFF" `
    --highlight-color "#FF4444" `
    --outline-width 6 `
    --position center
Write-Host ""

# Test 3: Word-by-word, large centered
Write-Host "[Test 3] Word mode - one word at a time, large (130px), cyan highlight" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_03_word_large.mp4 `
    --mode word `
    --font-size 130 `
    --font-color "#FFFFFF" `
    --highlight-color "#00E5FF" `
    --outline-width 7 `
    --position center
Write-Host ""

# Test 4: Chunk mode, 4 words at a time
Write-Host "[Test 4] Chunk mode - 4 words/chunk, large font (110px)" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_04_chunk_4words.mp4 `
    --mode chunk `
    --words-per-chunk 4 `
    --font-size 110 `
    --font-color "#FFFFFF" `
    --highlight-color "#FFD700" `
    --position lower
Write-Host ""

# Test 5: Karaoke, uppercase, 4 words/line, green highlight
Write-Host "[Test 5] Karaoke - UPPERCASE, 4 words/line, green highlight, upper position" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_05_karaoke_upper_green.mp4 `
    --mode karaoke `
    --words-per-line 4 `
    --max-lines 3 `
    --font-size 100 `
    --font-color "#FFFFFF" `
    --highlight-color "#00FF88" `
    --outline-width 5 `
    --uppercase `
    --position upper
Write-Host ""

# Test 6: TikTok preset
Write-Host "[Test 6] TikTok preset (all defaults from preset)" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_06_preset_tiktok.mp4 `
    --preset tiktok
Write-Host ""

# Test 7: Karaoke, no highlight, large clean white
Write-Host "[Test 7] Karaoke - no highlight, large clean white (120px)" -ForegroundColor Yellow
python -m killer_subtitles Timeline.mp4 -o test_07_no_highlight.mp4 `
    --mode karaoke `
    --words-per-line 4 `
    --max-lines 3 `
    --font-size 120 `
    --font-color "#FFFFFF" `
    --no-highlight `
    --position lower
Write-Host ""

Write-Host "=== All tests complete ===" -ForegroundColor Green
Write-Host "Output files:"
Get-ChildItem test_*.mp4 | ForEach-Object { Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB, 1)) MB" }
