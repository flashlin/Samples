set -e
source ./common.sh

# if isFileExists /usr/share/autojump/autojump.sh; then
#     echo "EXISTS"
# fi

if ! isFileContains ~/.bashrc 'PS1="\\w\\n\$ "'; then
    echo "setup bash shell newline"
    echo 'PS1="\w\n$ "' >> ~/.bashrc
    source ~/.bashrc
fi
