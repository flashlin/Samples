#!/bin/bash
set -e

sudo apt install wget git curl vim -y
sudo apt install zsh -y

echo 'download Nerd font..'
wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Regular.ttf &&
wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Bold.ttf  &&
wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Italic.ttf  &&
wget https://github.com/romkatv/powerlevel10k-media/raw/master/MesloLGS%20NF%20Bold%20Italic.ttf

echo 'install Nerd font...'
sudo cp ttf/*.ttf /usr/share/fonts/truetype/
sudo apt install fontconfig
echo 'refresh font cache...'
fc-cache -fv

echo 'install Oh-My-Zsh'
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

echo 'switch to zsh'
chsh -s $(which zsh)
echo $SHELL

echo 'install powerlevel10k theme'
git clone https://github.com/romkatv/powerlevel10k.git $ZSH_CUSTOM/themes/powerlevel10k
# zsh-autosuggestions自动提示插件
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
# zsh-syntax-highlighting语法高亮插件
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

# 配置powerlevel10k
p10k configure

echo 'vim ~/.zshrc'
echo '# 修改主题'
echo 'ZSH_THEME="powerlevel10k/powerlevel10k"'
echo '# 启用插件'
echo 'plugins=('
echo ' git'
echo ' zsh-autosuggestions'
echo ' zsh-syntax-highlighting'
echo ')'
