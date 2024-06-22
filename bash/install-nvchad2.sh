# mkdir -p ~/bin
# cp setup-files/win32yank.exe ~/bin/
# chmod +x ~/bin/win32yank.exe
# if grep -q 'export PATH=.*\$HOME/bin' ~/.bashrc; then
#     echo "$HOME/bin already in PATH. No changes made."
# else
#     # 如果沒有找到，添加新的 export PATH 行
#     echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
#     echo "Added $HOME/bin to PATH in ~/.bashrc"
# fi

##########
echo "install neovim ..."
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
sudo rm -rf /opt/nvim
sudo tar -C /opt -xzf nvim-linux64.tar.gz


sudo apt update
sudo apt install ninja-build


##########
echo "install lua-language-server ..."
mkdir -p ~/.config/lsp
cd ~/.config/lsp
git clone --depth=1 https://github.com/LuaLS/lua-language-server
cd lua-language-server
git submodule update --init --recursive
cd 3rd/luamake
compile/install.sh
cd ../..
./3rd/luamake/luamake rebuild
if grep -q 'export PATH="${HOME}/.config/lsp/lua-language-server/bin:${PATH}"' ~/.bashrc; then
    echo "lua-language-server/bin already in PATH. No changes made."
else
    echo 'export PATH="${HOME}/.config/lsp/lua-language-server/bin:${PATH}"' >> ~/.bashrc
    echo "Added lua-language-server/bin to PATH in ~/.bashrc"
fi

# curl -sLo/tmp/clipboard-provider https://github.com/nullchilly/clipboard-provider/releases/latest/download/clipboard-provider
# sudo mv /tmp/clipboard-provider /usr/local/bin/
# sudo chmod +x /usr/local/bin/clipboard-provider


git clone https://github.com/neovim/neovim.git
cd neovim
#git checkout v0.9.5
git checkout v0.10.0
sudo cp -r runtime /usr/share/nvim/
sudo cp -r runtime /usr/local/share/nvim
cd ..


./install-nvchad-config.sh