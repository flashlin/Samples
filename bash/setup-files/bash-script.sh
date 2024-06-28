########## bash scripts lit
ldd() {
    local pattern="$1"
    find . -type d \
    -not \( -path '*/\.git*' -prune \) \
    -not \( -path '*/node_modules*' -prune \) \
    -regextype posix-extended -iregex ".*${pattern}.*" -print0 | 
    while IFS= read -r -d '' dir; do
        du -sh "$dir"
    done
}
alias ldd='ldd'


ld() {
    local pattern="$1"
    find . -mindepth 1 -maxdepth 1 -type d \
    -not \( -name '\.git' \) \
    -not \( -name 'node_modules' \) \
    -not \( -name 'bin' \) \
    -regextype posix-extended -iregex ".*${pattern}.*" -print0 | 
    while IFS= read -r -d '' dir; do
        du -sh "$dir"
    done
}
alias ld='ld'

# Fuzzy search Git branches in a repo
# Looks for local and remote branches
gtb() {
    local pattern=$*
        local branches branch
        branches=$(git branch --all | awk 'tolower($0) ~ /'"$pattern"'/') &&
        branch=$(echo "$branches" | fzf-tmux -p --reverse -1 -0 +m) &&
        if [ "$branch" = "" ]; then
            echo "[$0] No branch matches the provided pattern"; return;
    fi;
    git checkout "$(echo "$branch" | sed "s/.* //" | sed "s#remotes/[^/]*/##")"
}
alias gtb='gtb'


# Fuzzy search over Git commits
# Enter will view the commit
# Ctrl-o will checkout the selected commit
gtshow() {
  git log --graph --color=always \
      --format="%C(auto)%h%d %s %C(black)%C(bold)%cr" "$@" |
  fzf --ansi --no-sort --reverse --tiebreak=index --bind=ctrl-s:toggle-sort --preview \
         'f() { set -- $(echo -- "$@" | grep -o "[a-f0-9]\{7\}"); [ $# -eq 0 ] || git show --color=always $1 ; }; f {}' \
      --header "enter to view, ctrl-o to checkout" \
      --bind "q:abort,ctrl-f:preview-page-down,ctrl-b:preview-page-up" \
      --bind "ctrl-o:become:(echo {} | grep -o '[a-f0-9]\{7\}' | head -1 | xargs git checkout)" \
      --bind "ctrl-m:execute:
                (grep -o '[a-f0-9]\{7\}' | head -1 |
                xargs -I % sh -c 'git show --color=always % | less -R') << 'FZF-EOF'
                {}
FZF-EOF" --preview-window=right:60%
}
alias gtshow='gtshow'
########## end