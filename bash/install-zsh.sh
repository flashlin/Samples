sudo apt install zsh
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"


git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
echo "請修改 ~/.zshrc 內容如下"
echo "# plugins=(git) -> plugins=(git zsh-autosuggestions)"

#./install-autojump.sh
echo "profile is in ~/.zshrc"


