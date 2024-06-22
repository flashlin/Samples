mkdir -p ~/bin
cp setup-files/win32yank.exe ~/bin/
chmod +x ~/bin/win32yank.exe
if grep -q 'export PATH=.*\$HOME/bin' ~/.bashrc; then
    echo "$HOME/bin already in PATH. No changes made."
else
    # 如果沒有找到，添加新的 export PATH 行
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    echo "Added $HOME/bin to PATH in ~/.bashrc"
fi


git clone https://github.com/neovim/neovim.git
cd neovim
#git checkout v0.9.5
git checkout v0.10.0
sudo cp -r runtime /usr/share/nvim/
sudo cp -r runtime /usr/local/share/nvim
cd ..


rm -rf ~/.local/share/nvim
rm -rf ~/.config/nvim
git clone https://github.com/NvChad/starter ~/.config/nvim && nvim

echo "複製自訂的按鍵綁定設定"
cp nvchad-files/mappings.lua ~/.config/nvim/lua/mappings.lua

echo "安裝 Telescope 的 live_grep 功能所需的依賴"
sudo apt-get install ripgrep

echo "如果你想修改 init 可按照下面指令"
echo "nvim ~/.config/nvim/init.lua"