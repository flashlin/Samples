#sudo add-apt-repository ppa:neovim-ppa/stable
#sudo apt-get update
#sudo apt-get install -y neovim
#sudo apt-get install -y python2-dev python2-pip python3-dev python3-pip


curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux64.tar.gz
sudo rm -rf /opt/nvim
sudo tar -C /opt -xzf nvim-linux64.tar.gz


check_path() {
   local target_path="$1"
   local contains_path=false

   # 使用 ":" 分割 PATH 字符串，並遍歷每個路徑
   IFS=":" read -ra path_array <<< "$PATH"

   for path in "${path_array[@]}"; do
        # 檢查是否包含目標路徑
        if [ "$path" == "$target_path" ]; then
            contains_path=true
            break
        fi
   done

   echo $contains_path
}

if [ $(check_path "/opt/nvim-linux64/bin") == false ]; then
   echo "export PATH=\$PATH:/opt/nvim-linux64/bin" >> ~/.bashrc
   source ~/.bashrc
fi


rm -rf ~/.config/nvim
rm -rf ~/.cache/nvim
rm -rf ~/.local/share/nvim

git clone https://github.com/NvChad/NvChad ~/.config/nvim --depth 1
NVCHAD_EXAMPLE_CONFIG=n nvim +'hi NormalFloat guibg=#1e222a' +PackerSync


echo "~/.config/nvim/lua/plugins/init.lua"
echo "~/.config/nvim/lua/custom/plugins.lua"