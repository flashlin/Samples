# 設定顏色
RED="\033[31m"   # 紅色
GREEN="\033[32m" # 綠色
NC="\033[0m"     # 取消顏色

# 檢查 result.trx 檔案是否存在
if [[ ! -f ./TestResults.trx ]]; then
    echo "檔案 result.trx 不存在."
    exit 1
fi

# 逐行讀取檔案內容
while IFS= read -r line; do
    if [[ "$line" == *"Error Message"* ]]; then
        echo -e "${RED}${line}${NC}"  # 顯示紅色
        continue  # 提早返回到迴圈開始
    fi
    
    if [[ "$line" == *"Failed!"* ]]; then
        echo -e "${RED}${line}${NC}"  # 顯示紅色
        continue  # 提早返回到迴圈開始
    fi
    
    if [[ "$line" == *"Passed!"* ]]; then
        echo -e "${GREEN}${line}${NC}"  # 顯示綠色
        continue  # 提早返回到迴圈開始
    fi

    echo "$line"  # 顯示正常顏色
done < ./TestResults.trx