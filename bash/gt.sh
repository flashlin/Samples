#!/bin/bash

# Print info message in green
info() {
  echo -e "\033[32m$1\033[0m"
}

# Print error message in red
show_error() {
  echo -e "\033[31m$1\033[0m"
}

# Execute command and print it in green
invoke_cmd() {
  echo -e "\033[32m$1\033[0m"
  eval "$1"
}

# Undo all uncommitted or unsaved changes
undo_all_uncommitted_or_unsaved_changes() {
  info "Undo all uncommitted or unsaved changes"
  invoke_cmd "git checkout -- ."
}

# Git init and copy .gitignore-template
init_git() {
  local script_dir=$(cd "$(dirname "$0")" && pwd)
  local current_dir=$(pwd)
  cp "$script_dir/.gitignore-template" "$current_dir/.gitignore"
  invoke_cmd "git init"
}

# Merge master to branch
merge_to_branch() {
  local branch="$1"
  info "取得遠端資料並更新本地 master 程式碼"
  git fetch origin master
  git checkout master
  git merge origin/master
  info "將 master 的程式碼合併至新分支中"
  git checkout "$branch"
  git merge master
}

# 強迫刪除 git repo 某資料夾(含歷史)
force_remove_folder() {
  local folder="$1"
  read -p "強迫刪除 git repo $folder 整個資料夾, 包含歷史紀錄 (yes/no)? " answer
  if [[ "$answer" != "yes" ]]; then
    return
  fi
  info "強迫刪除整個 $folder 資料夾包含歷史紀錄..."
  invoke_cmd "git filter-branch --force --index-filter 'git rm --cached -r --ignore-unmatch $folder/' --prune-empty --tag-name-filter cat -- --all"
  invoke_cmd "git push origin master --force --tags"
  rm -rf .git/refs/original/
  invoke_cmd "git reflog expire --expire=now --all"
  invoke_cmd "git gc --prune=now"
  invoke_cmd "git gc --aggressive --prune=now"
}

# Undo previous commit
undo_previous_commit() {
  info "復原上一個提交"
  invoke_cmd "git reset --hard HEAD~1"
}

# Remove added folder from git
remove_added_folder() {
  local folder="$1"
  if [[ -z "$folder" ]]; then
    show_error "gt rd <folder>"
    return
  fi
  invoke_cmd "git rm -r --cached $folder"
}

# Undo uncommitted files and clean
undo_uncommitted() {
  local specific_file="$1"
  if [[ -n "$specific_file" ]]; then
    invoke_cmd "git checkout -- $specific_file"
    invoke_cmd "git clean -f"
    return
  fi
  undo_all_uncommitted_or_unsaved_changes
  git status
}

# Revert file to specific commit
revert_file_to_commit() {
  local hash_id="$1"
  local file="$2"
  if [[ -z "$hash_id" ]]; then
    echo "Please input hash id"
    return
  fi
  if [[ -z "$file" ]]; then
    echo "Please input file name"
    return
  fi
  echo "針對指定 commit $hash_id 的特定檔案進行 revert"
  invoke_cmd "git checkout $hash_id^ -- $file"
  invoke_cmd "git status"
  invoke_cmd "git commit -m 'Revert $file to $hash_id version'"
}

# Abort commit files
abort_commit_files() {
  invoke_cmd "git reset HEAD~1"
  invoke_cmd "git status"
}

# Commit with message
commit_with_message() {
  local description="$1"
  invoke_cmd "git commit -m '$description'"
}

# Show commit info
show_commit_info() {
  local hash="$1"
  invoke_cmd "git show -s --format='fuller' $hash"
}

# Show short log
show_short_log() {
  invoke_cmd 'git log --pretty=format:"%h - %an, %cd : %s"'
}

# Show changed files in commit
show_commit_changed_files() {
  local hash_id="$1"
  echo "只列出該 commit 中變更的檔案名稱，而不顯示內容變更細節"
  invoke_cmd "git diff-tree --no-commit-id --name-only -r $hash_id"
}

# Show long log
show_long_log() {
  invoke_cmd "git log -p --format='fuller' --graph"
}

# List unpushed commits
list_unpushed_commits() {
  local name=$(git branch --show-current)
  invoke_cmd "git log --oneline origin/$name..HEAD"
}

# Checkout branch
checkout_branch() {
  local branch="$1"
  if [[ -z "$branch" ]]; then
    branch="master"
  fi
  invoke_cmd "git checkout $branch"
}

# Show diff for all modified files (side-by-side)
show_diff() {
  git diff --name-status | while IFS=$'\t' read -r status path; do
    case $status in
      M)
        echo -ne " M "
        echo " $path"
        staged_path="/tmp/__staged__.txt"
        git show ":$path" 2>/dev/null > "$staged_path"
        working_path="/tmp/__working__.txt"
        cat "$path" > "$working_path"
        diff -y --width=160 --suppress-common-lines "$staged_path" "$working_path"
        rm -f "$staged_path" "$working_path"
        ;;
      A)
        echo -ne " A "
        echo " $path"
        ;;
      D)
        echo -ne " D "
        echo " $path"
        ;;
      *)
        echo -ne " $status "
        echo " $path"
        ;;
    esac
  done
}

# Show diff with vimdiff
show_diff_vimdiff() {
  invoke_cmd "git difftool --tool=vimdiff -y"
}

# Pull and update submodules
pull_and_update_submodules() {
  invoke_cmd "git pull"
  invoke_cmd "git submodule sync"
  invoke_cmd "git submodule update --init --remote --recursive"
}

# Push
push() {
  invoke_cmd "git push"
}

# Add file or folder
add_file_or_folder() {
  local file_or_folder="$1"
  if [[ -z "$file_or_folder" ]]; then
    file_or_folder="."
  fi
  invoke_cmd "git add $file_or_folder"
  invoke_cmd "git status"
}

# Add all and commit and push
add_commit_push() {
  local description="$1"
  if [[ -z "$description" ]]; then
    show_error "gt am 'comment'"
    echo "please add comment description"
    return
  fi
  invoke_cmd "git add ."
  invoke_cmd "git commit -m '$description'"
  invoke_cmd "git push"
}

# List all branches
list_all_branches() {
  invoke_cmd "git branch -a"
}

# Create branch
create_branch() {
  local branch_name="$1"
  invoke_cmd "git branch $branch_name"
}

# Stash save
stash_save() {
  local comment="$1"
  invoke_cmd "git stash save -u '$comment'"
}

# Stash pop
stash_pop() {
  invoke_cmd "git stash pop"
}

# Stash clear
stash_clear() {
  invoke_cmd "git stash clear"
}

# Stash list
stash_list() {
  info "列出 stash 清單"
  invoke_cmd "git stash list"
}

# Show status or show commit
show_status_or_commit() {
  local hash="$1"
  if [[ -n "$hash" ]]; then
    invoke_cmd "git show $hash"
    return
  fi
  invoke_cmd "git status"
}

# Show git repo size
show_info() {
  echo "顯示目前專案的Git倉庫所佔用的檔案空間"
  echo "'size'欄位加上'size-pack'欄位所顯示的值就是 Git 倉庫所佔用的檔案空間"
  invoke_cmd "git count-objects -vH"
}

# Apply .gitignore
apply_gitignore() {
  invoke_cmd "git rm -r --cached ."
  invoke_cmd "git add ."
  invoke_cmd "git commit -m 'update .gitignore'"
  invoke_cmd "git filter-branch --force"
  invoke_cmd "git push --force --all"
}

# Free up git space
freeup_git() {
  invoke_cmd "git reflog expire --expire=now --expire-unreachable=now --all"
  invoke_cmd "git gc --prune=all --aggressive"
}

# Remove file from committed files
remove_file() {
  local file="$1"
  echo "Please input '$file' in .gitignore file"
  invoke_cmd "git rm $file"
}

# Clean untracked files
clean_untracked() {
  info "刪除當前目錄下沒有被track過的檔案和資料夾"
  invoke_cmd "git clean -df"
}

# Show remote branches by commit date
show_remote_branches() {
  invoke_cmd "git for-each-ref --sort=-committerdate"
}

# Pull all submodules
pull_all_submodules() {
  invoke_cmd "git submodule foreach git pull"
}

# Abort all unstage files
abort_all_unstage_files() {
  invoke_cmd "git restore ."
}

# Main entry
case "$1" in
  init) init_git ;;
  mtb) merge_to_branch "$2" ;;
  frf) force_remove_folder "$2" ;;
  r) undo_previous_commit ;;
  rd) remove_added_folder "$2" ;;
  u) undo_uncommitted "$2" ;;
  uf) revert_file_to_commit "$2" "$3" ;;
  uc) abort_commit_files ;;
  m) commit_with_message "$2" ;;
  h) show_commit_info "$2" ;;
  l) show_short_log ;;
  lf) show_commit_changed_files "$2" ;;
  ll) show_long_log ;;
  lc) list_unpushed_commits ;;
  c) checkout_branch "$2" ;;
  df) show_diff ;;
  dff) show_diff_vimdiff ;;
  pl) pull_and_update_submodules ;;
  p) push ;;
  a) add_file_or_folder "$2" ;;
  am) add_commit_push "$2" ;;
  lb) list_all_branches ;;
  b) create_branch "$2" ;;
  st) stash_save "$2" ;;
  stp) stash_pop ;;
  stc) stash_clear ;;
  stl) stash_list ;;
  s) show_status_or_commit "$2" ;;
  info) show_info ;;
  apply-gitignore) apply_gitignore ;;
  freeup) freeup_git ;;
  rm) remove_file "$2" ;;
  cl) clean_untracked ;;
  sr) show_remote_branches ;;
  plm) pull_all_submodules ;;
  ab) abort_all_unstage_files ;;
  *)
    echo ""
    info "git helper by flash"
    echo ""
    echo "a <file or folder> :add unstage files"
    echo "ab                 :abort all unstage files"
    echo "am 'comment desc'  :add unstage files and commit"
    echo "b [branch name]    :create branch or show current branch when no [branch name]"
    echo "c <branch name>    :checkout branch, if branch name is empty, checkout master"
    echo "cl                 :clean untrack files"
    echo "df                 :diff side-by-side"
    echo "dff                :diff horizontal (vimdiff)"
    echo "frf <folder>       :force remove folder (include history) in git repo"
    echo "h <hash>           :show hash info"
    echo "init               :init and add default .gitignore"
    echo "info               :show git repo size"
    echo "l                  :show short log"
    echo "ll                 :show long log"
    echo "lf <hash>          :show hash changed files"
    echo "lc                 :list unpushed commit"
    echo "r                  :undo previous action"
    echo "rm <file>          :remove file in commited file"
    echo "rd <folder>        :remove add folder in git"
    echo "p                  :push"
    echo "pl                 :pull"
    echo "lb                 :list all branches"
    echo "m 'comment desc'   :add files and commit"
    echo "mtb <branch name>  :merge into branch"
    echo "s                  :show current status"
    echo "s <hash>           :show hash changed files"
    echo "sr                 :show remote branch names order by commited date desc"
    echo "st [comment]       :stash save"
    echo "stl                :stash list"
    echo "stp                :stash pop"
    echo "stc                :stash clear"
    echo "u [file]           :undo uncommitted files and clean"
    echo "uc                 :abort commit files"
    echo "uf <hashId> <file> :revert commit file for hashId"
    echo "plm                :pull all submodules"
    ;;
esac 