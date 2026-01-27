#!/bin/bash

if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository"
    exit 1
fi

has_main=$(git rev-parse --verify main > /dev/null 2>&1 && echo "yes" || echo "no")
has_master=$(git rev-parse --verify master > /dev/null 2>&1 && echo "yes" || echo "no")

if [[ "$has_main" == "no" && "$has_master" == "no" ]]; then
    echo "Error: Neither 'main' nor 'master' branch exists"
    exit 1
fi

if [[ "$has_main" == "yes" && "$has_master" == "yes" ]]; then
    read -p "Both 'main' and 'master' exist. Which base branch to use? [main]: " base_input
    if [ -z "$base_input" ]; then
        BASE_BRANCH="main"
    elif [[ "$base_input" == "main" || "$base_input" == "master" ]]; then
        BASE_BRANCH="$base_input"
    else
        echo "Error: Invalid input. Please enter 'main' or 'master'"
        exit 1
    fi
elif [[ "$has_main" == "yes" ]]; then
    BASE_BRANCH="main"
else
    BASE_BRANCH="master"
fi

echo "Using base branch: $BASE_BRANCH"
echo ""
echo "Select a branch to squash:"

selected_branch=$(git branch --format='%(refname:short)' | grep -v "^${BASE_BRANCH}$" | fzf --height=40% --reverse --prompt="Select branch > ")

if [ -z "$selected_branch" ]; then
    echo "No branch selected, exiting."
    exit 1
fi

echo "Selected branch: $selected_branch"
echo ""
echo "Enter the commit message for squashed commit (leave empty to cancel):"
read -r commit_message

if [ -z "$commit_message" ]; then
    echo "No commit message provided, cancelled."
    exit 1
fi

echo ""
echo "Switching to branch: $selected_branch"
git checkout "$selected_branch"

if [ $? -ne 0 ]; then
    echo "Error: Failed to checkout branch"
    exit 1
fi

echo ""
echo "Resetting to $BASE_BRANCH (soft)..."
git reset --soft "$BASE_BRANCH"

if [ $? -ne 0 ]; then
    echo "Error: Failed to reset"
    exit 1
fi

echo ""
echo "Creating squashed commit..."
git commit -m "$commit_message"

if [ $? -ne 0 ]; then
    echo "Error: Failed to commit"
    exit 1
fi

echo ""
echo "New commit:"
git log --oneline -n 1
echo ""

read -p "Force push to origin/$selected_branch? (y/N): " confirm_push

if [[ "$confirm_push" == "y" || "$confirm_push" == "Y" ]]; then
    echo "Force pushing..."
    git push origin "$selected_branch" --force

    if [ $? -eq 0 ]; then
        echo "Force push completed successfully"
    else
        echo "Error: Force push failed"
        exit 1
    fi
else
    echo "Push cancelled. You can manually push later with:"
    echo "  git push origin $selected_branch --force"
fi
