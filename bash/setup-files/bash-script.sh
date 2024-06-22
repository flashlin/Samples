########## bash scripts lit
ld() {
    local pattern="$1"
    find . -type d -regextype posix-extended -iregex ".*${pattern}.*" -print0 | 
    while IFS= read -r -d '' dir; do
        du -sh "$dir"
    done
}
alias ld='ld'
########## end