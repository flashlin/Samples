echo "複製自訂的按鍵綁定設定"
cp nvchad-files/init.lua ~/.config/nvim/init.lua
cp -r nvchad-files/lua ~/.config/nvim

echo "安裝 Telescope 的 live_grep 功能所需的依賴"
sudo apt-get install ripgrep

echo "如果你想修改 init 可按照下面指令"
echo "nvim ~/.config/nvim/init.lua"