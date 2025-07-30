# dk 開發指南

## 快速開始

### 使用 dev.sh 開發腳本

我們提供了一個便利的開發腳本 `dev.sh`，可以快速編譯和運行 dk 工具。

#### 基本用法

```bash
# 編譯並運行 dk（debug 模式）
./dev.sh

# 編譯並運行 dk 的幫助信息
./dev.sh -- --help

# 編譯並運行特定命令
./dev.sh -- logs
./dev.sh -- bash
./dev.sh -- rm
```

#### 選項

```bash
# Release 模式編譯（優化版本）
./dev.sh --release

# 清理之前的編譯結果
./dev.sh --clean

# 只檢查代碼，不編譯
./dev.sh --check

# 運行測試
./dev.sh --test

# 顯示幫助
./dev.sh --help
```

#### 組合使用

```bash
# 清理並重新編譯 release 版本
./dev.sh --clean --release

# Release 模式編譯並運行 logs 命令
./dev.sh --release -- logs

# 清理、編譯並運行
./dev.sh --clean -- --help
```

## 開發工作流程

### 1. 日常開發
```bash
# 快速測試修改
./dev.sh -- --help

# 測試特定功能
./dev.sh -- logs
```

### 2. 代碼檢查
```bash
# 檢查語法錯誤
./dev.sh --check

# 運行測試
./dev.sh --test

# 格式化代碼
cargo fmt

# 代碼檢查
cargo clippy
```

### 3. 發布準備
```bash
# 清理並編譯 release 版本
./dev.sh --clean --release

# 測試 release 版本
./dev.sh --release -- --help
```

## 項目結構

```
dk/
├── src/
│   ├── main.rs          # 主程式入口
│   ├── commands.rs      # 命令實作
│   ├── docker.rs        # Docker API 客戶端
│   ├── types.rs         # 資料結構
│   └── ui.rs           # 終端 UI
├── Cargo.toml          # 專案配置
├── dev.sh              # 開發腳本
├── install.sh          # 安裝腳本
└── README.md           # 使用說明
```

## 開發提示

### 快速測試
- 使用 `./dev.sh --check` 快速檢查語法
- 使用 `./dev.sh` 快速編譯和測試
- 修改代碼後直接運行 `./dev.sh` 即可看到效果

### 調試
- Debug 模式編譯包含調試信息，適合開發
- 可以在代碼中使用 `println!` 或 `log::debug!` 進行調試
- 使用 `RUST_LOG=debug ./target/debug/dk` 查看詳細日誌

### 性能測試
- 使用 `./dev.sh --release` 編譯優化版本
- Release 版本體積更小，運行更快

## 常見問題

### Q: 編譯錯誤
A: 運行 `./dev.sh --clean --check` 清理並檢查

### Q: Docker 連接問題
A: 確保 Docker 服務正在運行，並且當前用戶有權限訪問 Docker

### Q: 依賴問題
A: 運行 `cargo update` 更新依賴

## 貢獻指南

1. Fork 專案
2. 創建功能分支
3. 使用 `./dev.sh --check` 檢查代碼
4. 使用 `./dev.sh --test` 運行測試
5. 提交 Pull Request

## 發布流程

1. 更新版本號 (`Cargo.toml`)
2. 運行完整測試 `./dev.sh --test`
3. 編譯 release 版本 `./dev.sh --release`
4. 更新 CHANGELOG
5. 創建 Git tag
6. 發布到 crates.io (可選)