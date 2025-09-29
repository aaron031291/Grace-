# Grace Git Workflow Helper (PowerShell)
# Automates common Git operations for development across environments

param(
    [Parameter(Mandatory=$true)]
    [string]$Command,
    [string]$Arg1,
    [string]$Arg2,
    [string]$Arg3
)

# Colors for output
$colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
}

function Write-Status($message) {
    Write-Host "[INFO] $message" -ForegroundColor $colors.Blue
}

function Write-Success($message) {
    Write-Host "[SUCCESS] $message" -ForegroundColor $colors.Green
}

function Write-Warning($message) {
    Write-Host "[WARNING] $message" -ForegroundColor $colors.Yellow
}

function Write-Error($message) {
    Write-Host "[ERROR] $message" -ForegroundColor $colors.Red
}

# Check if git is available
function Test-Git {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Error "Git is not installed. Please install Git first."
        exit 1
    }
}

# Check if we're in a git repository
function Test-GitRepo {
    try {
        git rev-parse --git-dir | Out-Null
    } catch {
        Write-Error "Not in a Git repository. Please run this from the Grace- directory."
        exit 1
    }
}

# Setup Git configuration
function Set-GitConfig {
    Write-Status "Setting up Git configuration..."
    
    # Check if user name and email are set
    try {
        git config user.name | Out-Null
    } catch {
        $username = Read-Host "Enter your Git username"
        git config --global user.name $username
    }
    
    try {
        git config user.email | Out-Null
    } catch {
        $email = Read-Host "Enter your Git email"
        git config --global user.email $email
    }
    
    # Set useful Git defaults
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    git config --global push.default simple
    git config --global credential.helper manager
    
    Write-Success "Git configuration completed"
}

# Create new feature branch
function New-Branch {
    param([string]$BranchName)
    
    if (-not $BranchName) {
        Write-Error "Branch name is required"
        Write-Host "Usage: .\git-workflow.ps1 new-branch <branch-name>"
        exit 1
    }
    
    Write-Status "Creating new branch: $BranchName"
    
    # Make sure we're on main and up to date
    git checkout main
    git pull origin main
    
    # Create and switch to new branch
    git checkout -b $BranchName
    
    Write-Success "Created and switched to branch: $BranchName"
}

# Sync with main branch
function Sync-Main {
    Write-Status "Syncing with main branch..."
    
    $currentBranch = git branch --show-current
    
    # Switch to main and pull latest
    git checkout main
    git pull origin main
    
    # Switch back to original branch if it wasn't main
    if ($currentBranch -ne "main") {
        git checkout $currentBranch
        Write-Status "Merging latest changes from main..."
        git merge main
    }
    
    Write-Success "Sync completed"
}

# Commit changes with conventional commit format
function Commit-Changes {
    param(
        [string]$Type,
        [string]$Scope,
        [string]$Message
    )
    
    if (-not $Type -or -not $Message) {
        Write-Error "Type and message are required"
        Write-Host "Usage: .\git-workflow.ps1 commit <type> [scope] <message>"
        Write-Host "Types: feat, fix, docs, style, refactor, test, chore"
        Write-Host "Scopes: governance, mldl, audit, multi-os, etc."
        exit 1
    }
    
    # Build commit message
    $commitMsg = $Type
    if ($Scope -and $Scope -ne $Message) {
        $commitMsg = "$commitMsg($Scope)"
    }
    $commitMsg = "$commitMsg`: $Message"
    
    Write-Status "Committing changes with message: $commitMsg"
    
    # Add all changes and commit
    git add .
    git commit -m $commitMsg
    
    Write-Success "Changes committed successfully"
}

# Push changes to remote
function Push-Changes {
    Write-Status "Pushing changes to remote..."
    
    $currentBranch = git branch --show-current
    
    # Check if branch exists on remote
    try {
        git ls-remote --exit-code --heads origin $currentBranch | Out-Null
        git push
    } catch {
        Write-Status "Setting upstream for new branch..."
        git push -u origin $currentBranch
    }
    
    Write-Success "Changes pushed successfully"
}

# Create pull request (GitHub CLI required)
function New-PullRequest {
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        Write-Status "Creating pull request..."
        gh pr create --fill
        Write-Success "Pull request created"
    } else {
        Write-Warning "GitHub CLI not installed. Please create PR manually at:"
        $currentBranch = git branch --show-current
        Write-Host "https://github.com/aaron031291/Grace-/compare/main...$currentBranch"
    }
}

# Show status and helpful info
function Show-Status {
    Write-Status "Git Status:"
    git status --short
    Write-Host ""
    
    $currentBranch = git branch --show-current
    $remoteUrl = git remote get-url origin
    
    Write-Status "Current branch: $currentBranch"
    Write-Status "Remote URL: $remoteUrl"
    Write-Host ""
    
    $changes = git status --porcelain
    if ($changes) {
        Write-Warning "You have uncommitted changes"
    } else {
        Write-Success "Working directory clean"
    }
}

# Run tests before pushing
function Test-Grace {
    Write-Status "Running tests..."
    
    # Basic import test
    try {
        python -c "import grace; print('✅ Grace imports successfully')" | Out-Null
        Write-Success "Basic import test passed"
    } catch {
        Write-Error "Basic import test failed"
        return $false
    }
    
    # Run smoke tests if available
    if (Test-Path "demo_and_tests/tests/smoke_tests.py") {
        try {
            python demo_and_tests/tests/smoke_tests.py | Out-Null
            Write-Success "Smoke tests passed"
        } catch {
            Write-Error "Smoke tests failed"
            return $false
        }
    }
    
    Write-Success "All tests passed"
    return $true
}

# Complete workflow: test, commit, push
function Complete-Workflow {
    param(
        [string]$Type,
        [string]$Scope,
        [string]$Message
    )
    
    if (-not $Type -or -not $Message) {
        Write-Error "Type and message are required"
        Write-Host "Usage: .\git-workflow.ps1 workflow <type> [scope] <message>"
        exit 1
    }
    
    Write-Status "Running complete workflow..."
    
    # Run tests
    if (-not (Test-Grace)) {
        Write-Error "Tests failed. Aborting workflow."
        exit 1
    }
    
    # Commit changes
    Commit-Changes $Type $Scope $Message
    
    # Push changes
    Push-Changes
    
    Write-Success "Workflow completed successfully!"
}

# Main execution
Test-Git
Test-GitRepo

switch ($Command) {
    "setup" { Set-GitConfig }
    "new-branch" { New-Branch $Arg1 }
    "sync" { Sync-Main }
    "commit" { Commit-Changes $Arg1 $Arg2 $Arg3 }
    "push" { Push-Changes }
    "pr" { New-PullRequest }
    "status" { Show-Status }
    "test" { Test-Grace }
    "workflow" { Complete-Workflow $Arg1 $Arg2 $Arg3 }
    default {
        Write-Host "Grace Git Workflow Helper (PowerShell)"
        Write-Host ""
        Write-Host "Usage: .\git-workflow.ps1 <command> [options]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  setup              - Configure Git settings"
        Write-Host "  new-branch <name>  - Create new feature branch"
        Write-Host "  sync               - Sync with main branch"
        Write-Host "  commit <type> [scope] <msg> - Commit with conventional format"
        Write-Host "  push               - Push changes to remote"
        Write-Host "  pr                 - Create pull request (requires GitHub CLI)"
        Write-Host "  status             - Show Git status and info"
        Write-Host "  test               - Run tests"
        Write-Host "  workflow <type> [scope] <msg> - Complete workflow (test, commit, push)"
        Write-Host "  help               - Show this help"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\git-workflow.ps1 new-branch feature/add-validation"
        Write-Host "  .\git-workflow.ps1 commit feat governance 'add new constitutional validator'"
        Write-Host "  .\git-workflow.ps1 workflow fix mldl 'resolve specialist consensus timeout'"
    }
}