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
########## end