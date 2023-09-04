pip install --upgrade pynvim
sudo ln -s /usr/bin/python3 /usr/bin/python

sh -c 'curl -fLo "$[XDG DATA HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
