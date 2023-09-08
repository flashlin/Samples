# 檢查是否提供了路徑參數
if [ $# -ne 1 ]; then
    echo "Please provide a PATH, ex: \"/demo/sample\""
    exit 1
fi

newPath=$1
eval newPath="$newPath"

if [[ ":$PATH:" != *":$newPath:"* ]]; then
    echo "Adding $newPath to PATH..."
    echo 'export PATH="$newPath:$PATH"' >> ~/.bashrc
    source ~/.bashrc
fi

echo "done."