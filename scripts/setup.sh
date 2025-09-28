#!/bin/bash

# Grace Development Environment Setup Script
# Automatically sets up the development environment across different platforms

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "=================================="
    echo "Grace Development Environment Setup"
    echo "=================================="
    echo -e "${NC}"
}

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

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin)
            OS="macos"
            ;;
        Linux)
            OS="linux"
            if grep -qi ubuntu /etc/os-release 2>/dev/null; then
                DISTRO="ubuntu"
            elif grep -qi debian /etc/os-release 2>/dev/null; then
                DISTRO="debian"
            elif grep -qi fedora /etc/os-release 2>/dev/null; then
                DISTRO="fedora"
            elif grep -qi centos /etc/os-release 2>/dev/null; then
                DISTRO="centos"
            else
                DISTRO="linux"
            fi
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            OS="windows"
            ;;
        *)
            OS="unknown"
            ;;
    esac
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Python based on OS
install_python() {
    print_status "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON=python3
        print_success "Python3 found: $(python3 --version)"
        return 0
    elif command_exists python; then
        if python --version 2>&1 | grep -q "Python 3"; then
            PYTHON=python
            print_success "Python found: $(python --version)"
            return 0
        fi
    fi
    
    print_warning "Python 3.11+ not found. Installing Python..."
    
    case $OS in
        "macos")
            if command_exists brew; then
                brew install python
            else
                print_error "Please install Homebrew first: https://brew.sh/"
                exit 1
            fi
            ;;
        "linux")
            case $DISTRO in
                "ubuntu"|"debian")
                    sudo apt update
                    sudo apt install -y python3 python3-pip python3-venv
                    ;;
                "fedora")
                    sudo dnf install -y python3 python3-pip
                    ;;
                "centos")
                    sudo yum install -y python3 python3-pip
                    ;;
                *)
                    print_error "Please install Python 3.11+ manually for your distribution"
                    exit 1
                    ;;
            esac
            ;;
        "windows")
            print_error "Please install Python from https://python.org/downloads/"
            exit 1
            ;;
        *)
            print_error "Unsupported operating system. Please install Python 3.11+ manually."
            exit 1
            ;;
    esac
}

# Install Git based on OS
install_git() {
    print_status "Checking Git installation..."
    
    if command_exists git; then
        print_success "Git found: $(git --version)"
        return 0
    fi
    
    print_warning "Git not found. Installing Git..."
    
    case $OS in
        "macos")
            if command_exists brew; then
                brew install git
            else
                print_error "Please install Git from https://git-scm.com/"
                exit 1
            fi
            ;;
        "linux")
            case $DISTRO in
                "ubuntu"|"debian")
                    sudo apt update
                    sudo apt install -y git
                    ;;
                "fedora")
                    sudo dnf install -y git
                    ;;
                "centos")
                    sudo yum install -y git
                    ;;
                *)
                    print_error "Please install Git manually for your distribution"
                    exit 1
                    ;;
            esac
            ;;
        "windows")
            print_error "Please install Git from https://git-scm.com/"
            exit 1
            ;;
        *)
            print_error "Please install Git manually for your operating system"
            exit 1
            ;;
    esac
}

# Setup Python virtual environment
setup_venv() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove existing venv and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_success "Using existing virtual environment"
            return 0
        fi
    fi
    
    $PYTHON -m venv venv
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        VENV_ACTIVATED=1
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        VENV_ACTIVATED=1
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
    
    print_success "Virtual environment created and activated"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed successfully"
}

# Setup environment configuration
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_success "Created .env from template"
            print_warning "Please edit .env file with your configuration"
        else
            print_warning ".env.template not found, skipping environment setup"
        fi
    else
        print_success ".env already exists"
    fi
}

# Setup Git hooks
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    if [ -f "scripts/pre-commit" ]; then
        if [ -d ".git/hooks" ]; then
            cp scripts/pre-commit .git/hooks/
            chmod +x .git/hooks/pre-commit
            print_success "Pre-commit hook installed"
        else
            print_warning "Not in a Git repository, skipping Git hooks"
        fi
    else
        print_warning "Pre-commit hook not found, skipping"
    fi
}

# Setup Git configuration
setup_git_config() {
    print_status "Setting up Git configuration..."
    
    # Check if Git user is configured
    if ! git config user.name >/dev/null 2>&1; then
        print_warning "Git user.name not configured"
        read -p "Enter your Git username: " git_username
        git config --global user.name "$git_username"
    fi
    
    if ! git config user.email >/dev/null 2>&1; then
        print_warning "Git user.email not configured"
        read -p "Enter your Git email: " git_email
        git config --global user.email "$git_email"
    fi
    
    # Set useful defaults
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    git config --global push.default simple
    
    print_success "Git configuration completed"
}

# Verify installation
verify_setup() {
    print_status "Verifying installation..."
    
    # Test Python import
    if $PYTHON -c "import grace; print('Grace imports successfully')" >/dev/null 2>&1; then
        print_success "Grace import test passed"
    else
        print_error "Grace import test failed"
        return 1
    fi
    
    # Test basic functionality
    if [ -f "demo_and_tests/tests/smoke_tests.py" ]; then
        if $PYTHON demo_and_tests/tests/smoke_tests.py >/dev/null 2>&1; then
            print_success "Basic functionality test passed"
        else
            print_warning "Basic functionality test failed (this may be expected in some environments)"
        fi
    fi
    
    print_success "Setup verification completed"
}

# Show next steps
show_next_steps() {
    echo
    print_success "Setup completed! 🎉"
    echo
    echo "Next steps:"
    echo "1. Activate virtual environment:"
    
    if [ "$OS" = "windows" ]; then
        echo "   venv\\Scripts\\activate"
    else
        echo "   source venv/bin/activate"
    fi
    
    echo "2. Edit .env file with your configuration"
    echo "3. Read DEVELOPMENT_SETUP.md for detailed development guide"
    echo "4. Try running: python -c \"import grace; print('Hello Grace!')\""
    echo "5. Use scripts/git-workflow.sh for Git operations"
    echo
    echo "For help: ./scripts/git-workflow.sh help"
}

# Main execution
main() {
    print_header
    
    detect_os
    print_status "Detected OS: $OS $([ -n "$DISTRO" ] && echo "($DISTRO)")"
    
    install_git
    install_python
    
    # Ensure we're in the right directory
    if [ ! -f "requirements.txt" ] || [ ! -d "grace" ]; then
        print_error "Not in Grace repository root directory"
        exit 1
    fi
    
    setup_venv
    install_dependencies
    setup_env
    setup_git_hooks
    setup_git_config
    verify_setup
    show_next_steps
}

# Check if script is being sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi