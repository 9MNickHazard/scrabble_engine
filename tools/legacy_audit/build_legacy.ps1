<#
Build the legacy Rust crate (legacy\scrabble_valid_moves) outside the repo and
produce importable .pyd copies for the audit scripts.

  powershell -ExecutionPolicy Bypass -File tools\legacy_audit\build_legacy.ps1 -Scratch <dir> [-Repo C:\dev\scrabble-engine]

Everything is written under -Scratch (crate copies, CARGO_TARGET_DIR, logs, .pyd
folders). Nothing under the repo is modified. Steps:

 1. untouched copy: cargo build --release --locked        (lib + bin, as shipped)
 2. untouched copy: cargo build --release --locked --lib  (lib only)
 3. probe copy with crate-type ["cdylib","rlib"]: build the bin only, to see the
    errors that remain once the library is linkable from main.rs
 4. diagnostic copies (ATTRIBUTION ONLY, never used for the headline numbers):
      noprune    : could_form_valid_word() always returns true
      nomaxspace : get_max_space() always returns BOARD_SIZE
      both       : both changes
 5. copy each built DLL to <Scratch>\pyd_<name>\scrabble_valid_moves_rust.pyd
    (a PyO3 extension module on Windows is just the DLL renamed to .pyd)

Python used by PyO3's build script: whatever `python` is on PATH, or set
$env:PYO3_PYTHON. PyO3 0.21 supports up to Python 3.12.
#>
param(
    [Parameter(Mandatory = $true)][string]$Scratch,
    [string]$Repo = "C:\dev\scrabble-engine"
)
$ErrorActionPreference = "Stop"
$src = Join-Path $Repo "legacy\scrabble_valid_moves"
$logs = Join-Path $Scratch "logs"
New-Item -ItemType Directory -Force $logs | Out-Null

function Copy-Crate([string]$name) {
    $dst = Join-Path $Scratch $name
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse $src $dst
    return $dst
}

function Invoke-Cargo([string]$crate, [string]$target, [string]$log, [string]$cargoArgs) {
    # cmd.exe redirection keeps the log byte-for-byte what cargo printed
    $env:CARGO_TARGET_DIR = $target
    Push-Location $crate
    try {
        cmd /c "cargo $cargoArgs > `"$log`" 2>&1"
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Write-Host ("[{0}] cargo {1} -> exit {2}  (log: {3})" -f (Split-Path $crate -Leaf), $cargoArgs, $code, $log)
}

function Set-Text([string]$path, [string]$old, [string]$new) {
    $t = [IO.File]::ReadAllText($path)
    if (-not $t.Contains($old)) { throw "patch anchor not found in ${path}: $old" }
    [IO.File]::WriteAllText($path, $t.Replace($old, $new))
}

function Save-Pyd([string]$target, [string]$name) {
    $dir = Join-Path $Scratch "pyd_$name"
    New-Item -ItemType Directory -Force $dir | Out-Null
    $dll = Join-Path $target "release\scrabble_valid_moves_rust.dll"
    $pyd = Join-Path $dir "scrabble_valid_moves_rust.pyd"
    $h = (Get-FileHash $dll -Algorithm SHA256).Hash
    if ((Test-Path $pyd) -and (Get-FileHash $pyd -Algorithm SHA256).Hash -eq $h) {
        Write-Host "  $pyd already up to date (SHA256 $h)"
    } else {
        Copy-Item $dll $pyd -Force
        Write-Host "  copied $dll -> $pyd (SHA256 $h)"
    }
}

# 1-2: untouched
$c = Copy-Crate "crate_untouched"
$t = Join-Path $Scratch "target_untouched"
Invoke-Cargo $c $t (Join-Path $logs "untouched_lib_and_bin.log") "build --release --locked"
Invoke-Cargo $c $t (Join-Path $logs "untouched_lib_only.log") "build --release --locked --lib"
Save-Pyd $t "untouched"

# 3: rlib probe
$c = Copy-Crate "crate_rlib_probe"
Set-Text (Join-Path $c "Cargo.toml") 'crate-type = ["cdylib"]' 'crate-type = ["cdylib", "rlib"]'
Invoke-Cargo $c (Join-Path $Scratch "target_rlib_probe") (Join-Path $logs "rlib_probe_bin.log") "build --release --locked --bin scrabble_valid_moves_rust"

# 4-5: diagnostic variants (shared target dir; the DLL is copied out after each build)
$prune_old = 'self.dictionary.prefixes.contains(&prefix.to_lowercase())'
$prune_new = '{ let _ = prefix; true } // AUDIT DIAGNOSTIC: prefix pruning disabled'
$max_old = 'fn get_max_space(&self, anchor: &str, direction: Direction) -> usize {'
$max_new = $max_old + "`n        if true { return BOARD_SIZE; } // AUDIT DIAGNOSTIC: no max_space limit"
$td = Join-Path $Scratch "target_diag"
foreach ($v in @("noprune", "nomaxspace", "both")) {
    $c = Copy-Crate "crate_diag_$v"
    $lib = Join-Path $c "src\lib.rs"
    if ($v -ne "nomaxspace") { Set-Text $lib $prune_old $prune_new }
    if ($v -ne "noprune") { Set-Text $lib $max_old $max_new }
    Invoke-Cargo $c $td (Join-Path $logs "diag_$v.log") "build --release --locked --lib"
    Save-Pyd $td $v
}
