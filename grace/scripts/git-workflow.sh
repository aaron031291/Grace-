#!/bin/bash

# Grace Git Workflow Helper
# Automates common Git operations for development across environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if git is available
check_git() {
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
}

# Check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a Git repository. Please run this from the Grace- directory."
        exit 1
    fi
}

# Setup Git configuration
setup_git() {
    print_status "Setting up Git configuration..."
    
    # Check if user name and email are set
    if ! git config user.name > /dev/null 2>&1; then
        read -p "Enter your Git username: " username
        git config --global user.name "$username"
    fi
    
    if ! git config user.email > /dev/null 2>&1; then
        read -p "Enter your Git email: " email
        git config --global user.email "$email"
    fi
    
    # Set useful Git defaults
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    git config --global push.default simple
    
    # Set up credential helper based on OS
    case "$(uname -s)" in
        Darwin)
            git config --global credential.helper osxkeychain
            ;;
        Linux)
            git config --global credential.helper store
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            git config --global credential.helper manager
            ;;
    esac
    
    print_success "Git configuration completed"
}

# Create new feature branch
new_branch() {
    local branch_name="$1"
    
    if [ -z "$branch_name" ]; then
        print_error "Branch name is required"
        echo "Usage: $0 new-branch <branch-name>"
        exit 1
    fi
    
    print_status "Creating new branch: $branch_name"
    
    # Make sure we're on main and up to date
    git checkout main
    git pull origin main
    
    # Create and switch to new branch
    git checkout -b "$branch_name"
    
    print_success "Created and switched to branch: $branch_name"
}

# Sync with main branch
sync_main() {
    print_status "Syncing with main branch..."
    
    local current_branch=$(git branch --show-current)
    
    # Switch to main and pull latest
    git checkout main
    git pull origin main
    
    # Switch back to original branch if it wasn't main
    if [ "$current_branch" != "main" ]; then
        git checkout "$current_branch"
        print_status "Merging latest changes from main..."
        git merge main
    fi
    
    print_success "Sync completed"
}

# Commit changes with conventional commit format
commit_changes() {
    local type="$1"
    local scope="$2"
    local message="$3"
    
    if [ -z "$type" ] || [ -z "$message" ]; then
        print_error "Type and message are required"
        echo "Usage: $0 commit <type> [scope] <message>"
        echo "Types: feat, fix, docs, style, refactor, test, chore"
        echo "Scopes: governance, mldl, audit, multi-os, etc."
        exit 1
    fi
    
    # Build commit message
    local commit_msg="$type"
    if [ -n "$scope" ] && [ "$scope" != "$message" ]; then
        commit_msg="$commit_msg($scope)"
    fi
    commit_msg="$commit_msg: $message"
    
    print_status "Committing changes with message: $commit_msg"
    
    # Add all changes and commit
    git add .
    git commit -m "$commit_msg"
    
    print_success "Changes committed successfully"
}

# Push changes to remote
push_changes() {
    print_status "Pushing changes to remote..."
    
    local current_branch=$(git branch --show-current)
    
    # Check if branch exists on remote
    if git ls-remote --exit-code --heads origin "$current_branch" > /dev/null 2>&1; then
        git push
    else
        print_status "Setting upstream for new branch..."
        git push -u origin "$current_branch"
    fi
    
    print_success "Changes pushed successfully"
}

# Create pull request (GitHub CLI required)
create_pr() {
    if command -v gh &> /dev/null; then
        print_status "Creating pull request..."
        gh pr create --fill
        print_success "Pull request created"
    else
        print_warning "GitHub CLI not installed. Please create PR manually at:"
        echo "https://github.com/aaron031291/Grace-/compare/main...$(git branch --show-current)"
    fi
}

# Show status and helpful info
show_status() {
    print_status "Git Status:"
    git status --short
    echo
    
    print_status "Current branch: $(git branch --show-current)"
    print_status "Remote URL: $(git remote get-url origin)"
    echo
    
    if git status --porcelain | grep -q .; then
        print_warning "You have uncommitted changes"
    else
        print_success "Working directory clean"
    fi
}

# Run tests before pushing
run_tests() {
    print_status "Running tests..."
    
    # Basic import test
    if python -c "import grace; print('✅ Grace imports successfully')" 2>/dev/null; then
        print_success "Basic import test passed"
    else
        print_error "Basic import test failed"
        return 1
    fi
    
    # Run smoke tests if available
    if [ -f "demo_and_tests/tests/smoke_tests.py" ]; then
        if python demo_and_tests/tests/smoke_tests.py; then
            print_success "Smoke tests passed"
        else
            print_error "Smoke tests failed"
            return 1
        fi
    fi
    
    print_success "All tests passed"
}

# Complete workflow: test, commit, push
complete_workflow() {
    local type="$1"
    local scope="$2"
    local message="$3"
    
    if [ -z "$type" ] || [ -z "$message" ]; then
        print_error "Type and message are required"
        echo "Usage: $0 workflow <type> [scope] <message>"
        exit 1
    fi
    
    print_status "Running complete workflow..."
    
    # Run tests
    if ! run_tests; then
        print_error "Tests failed. Aborting workflow."
        exit 1
    fi
    
    # Commit changes
    commit_changes "$type" "$scope" "$message"
    
    # Push changes
    push_changes
    
    print_success "Workflow completed successfully!"
}

# Main function
main() {
    check_git
    check_git_repo
    
    case "${1:-help}" in
        "setup")
            setup_git
            ;;
        "new-branch")
            new_branch "$2"
            ;;
        "sync")
            sync_main
            ;;
        "commit")
            commit_changes "$2" "$3" "$4"
            ;;
        "push")
            push_changes
            ;;
        "pr")
            create_pr
            ;;
        "status")
            show_status
            ;;
        "test")
            run_tests
            ;;
        "workflow")
            complete_workflow "$2" "$3" "$4"
            ;;
        "help"|*)
            echo "Grace Git Workflow Helper"
            echo ""
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  setup              - Configure Git settings"
            echo "  new-branch <name>  - Create new feature branch"
            echo "  sync               - Sync with main branch"
            echo "  commit <type> [scope] <msg> - Commit with conventional format"
            echo "  push               - Push changes to remote"
            echo "  pr                 - Create pull request (requires GitHub CLI)"
            echo "  status             - Show Git status and info"
            echo "  test               - Run tests"
            echo "  workflow <type> [scope] <msg> - Complete workflow (test, commit, push)"
            echo "  help               - Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 new-branch feature/add-validation"
            echo "  $0 commit feat governance 'add new constitutional validator'"
            echo "  $0 workflow fix mldl 'resolve specialist consensus timeout'"
            ;;
    esac
}

# Run main function with all arguments
main "$@"