Create a new commit for all of our uncommitted changes
run git status && git diff HEAD && git status --porcelain to see what files are uncommitted
add the untracked and changed files

Add an atomic commit message with an appropriate message

add a tag such as "feat", "fix", "docs", etc. that reflects our work

IMPORTANT:
- Commit message MUST be in English
- Do NOT add author information (Co-Authored-By)
- Do NOT add "Generated with AI" or similar attribution