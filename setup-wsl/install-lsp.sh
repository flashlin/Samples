# sudo apt update
# sudo apt install lua5.1 luarocks
# sudo luarocks install luarocks

# echo ""
# echo "install lua-language..."
# sudo luarocks install --server=https://luarocks.org/dev lua-language-server --check-lua-versions

echo ""
echo "install typescript..."
npm install -g typescript-language-server typescript
echo "Running the language server"
# npx typescript-language-server --stdio