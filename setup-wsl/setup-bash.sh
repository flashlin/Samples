set -e
source ./common.sh

#
if ! isFileContains ~/.bashrc '\\]\\n\\$ "'; then
    echo "setup bash shell newline"
    content='PS1="${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\n\$ "'
    echo $content >> ~/.bashrc
    source ~/.bashrc
fi
