# lf - File Finder

一個用 Rust 編寫的檔案搜尋工具，根據 regex 模式搜尋檔案名稱。

## 功能特色

- 支援 regex 模式匹配檔案名稱
- 遞迴搜尋子目錄
- 即時顯示搜尋進度
- 符合條件的檔案以綠色顯示
- 自動處理終端機寬度，長路徑會被截斷顯示

## 使用方法

```bash
cargo run -- "<regex_pattern>"
```

### 範例

搜尋所有 .txt 檔案：
```bash
cargo run -- ".*\.txt$"
```

搜尋以 "test" 開頭的檔案：
```bash
cargo run -- "^test.*"
```

搜尋包含 "config" 的檔案：
```bash
cargo run -- ".*config.*"
```

## 編譯

```bash
cargo build --release
```

編譯後的執行檔位於 `target/release/lf`

## 依賴項

- `regex`: 正則表達式支援
- `walkdir`: 遞迴目錄遍歷
- `colored`: 終端機顏色輸出
- `term_size`: 獲取終端機尺寸 