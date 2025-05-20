# sudo apt install icdiff

# Get the latest 20 commits and let user select one via fzf
select_commit() {
  # Use git log to get hash, author, date, and message, then pipe to fzf
  git log -20 --pretty=format:"%C(yellow)%h %C(green)%an %C(blue)%ad %C(reset)%s" --date=short |
    fzf --ansi --prompt="Select a commit: "
}

# Parse the selected line to extract the commit hash
get_commit_id() {
  local selected_line="$1"
  # The hash is the first field
  echo "$selected_line" | awk '{print $1}'
}

# Get modified files in the given commit
get_modified_files() {
  local commit_id="$1"
  # 只取出修改(M)的檔案名稱
  git diff-tree --no-commit-id --diff-filter=M --name-only -r "$commit_id"
}

# Show diff for each modified file using icdiff
show_icdiff_for_files() {
  local commit_id="$1"
  local files=("$@")
  # 從第二個參數開始才是檔案
  for file in "${files[@]:1}"; do
    echo "🔍 比較檔案：$file"
    icdiff -N <(git show "${commit_id}^":"$file") <(git show "${commit_id}":"$file")
    echo ""
  done
}

# Main flow
selected_line=$(select_commit)
commit_id=$(get_commit_id "$selected_line")

# For debug: print the selected commit id
echo "Selected commit id: $commit_id"

# Main flow
modified_files=($(get_modified_files "$commit_id"))
if [ ${#modified_files[@]} -eq 0 ]; then
  echo "No modified files in commit $commit_id"
else
  show_icdiff_for_files "$commit_id" "${modified_files[@]}"
fi

