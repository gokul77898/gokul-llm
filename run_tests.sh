#!/bin/bash

# Legal AI System - Test Runner
# This script runs all tests with various options

set -e

echo "=================================="
echo "Legal AI System - Test Runner"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
        exit 1
    fi
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest is not installed. Installing...${NC}"
    pip install pytest pytest-cov
fi

# Parse command line arguments
COVERAGE=false
VERBOSE=false
SPECIFIC=""
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --test|-t)
            SPECIFIC="$2"
            shift 2
            ;;
        --skip-gpu)
            MARKERS="-m 'not gpu'"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage     Generate coverage report"
            echo "  -v, --verbose      Verbose output"
            echo "  -t, --test FILE    Run specific test file"
            echo "  --skip-gpu         Skip GPU-dependent tests"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests"
            echo "  ./run_tests.sh --coverage         # Run with coverage"
            echo "  ./run_tests.sh -t test_mamba.py   # Run specific test"
            echo "  ./run_tests.sh --skip-gpu         # Skip GPU tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ -n "$SPECIFIC" ]; then
    PYTEST_CMD="$PYTEST_CMD tests/$SPECIFIC"
else
    PYTEST_CMD="$PYTEST_CMD tests/"
fi

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v -s"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

echo "Running: $PYTEST_CMD"
echo ""

# Run tests
$PYTEST_CMD
TEST_RESULT=$?

echo ""
echo "=================================="

if [ $TEST_RESULT -eq 0 ]; then
    print_status 0 "All tests passed!"
    
    if [ "$COVERAGE" = true ]; then
        echo ""
        echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
        echo "Open it with: open htmlcov/index.html"
    fi
else
    print_status 1 "Some tests failed!"
fi

echo "=================================="
exit $TEST_RESULT
