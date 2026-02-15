# Local CI Deploy

以 [gitlab-ci-local](https://github.com/firecow/gitlab-ci-local) 為基礎的輕量包裝，用於在本機執行 GitLab CI jobs。

## 安裝

```bash
./local-ci/setup.sh
```

## 使用方式

在包含 `.gitlab-ci.yml` 的專案目錄中執行：

```bash
cd b2c-support-agent

# 互動模式 - 從列表中選擇要執行的 job
../local-ci/local-ci-deploy.sh

# 列出所有可用的 jobs
../local-ci/local-ci-deploy.sh --list

# 直接執行指定的 job
../local-ci/local-ci-deploy.sh --job deploy

# Dry run - 顯示將執行的指令但不實際執行
../local-ci/local-ci-deploy.sh --dry-run
```

## 專案變數設定（.gitlab-ci-local-variables.yml）

- 在專案根目錄建立 `.gitlab-ci-local-variables.yml`，用來覆寫 `.gitlab-ci.yml` 中定義的變數
- `local-ci-deploy.sh` 會自動偵測並載入此檔案
- 可從範本複製後修改：

```bash
cp ../local-ci/templates/gitlab-ci-local-variables.yml .gitlab-ci-local-variables.yml
```

一般變數範例：

```yaml
DEPARTMENT: twdc
NAMESPACE: twdc
IMAGE_TAG: local
```

### File-type 變數

`gitlab-ci-local` 支援 file-type 變數。設定後，工具會將 `values` 的內容寫入暫存檔，並將變數值設為該暫存檔的路徑。適用於 CI 中需要以檔案形式提供的變數（如 `.env` 檔案內容）。

格式如下：

```yaml
ENV_FILE:
  type: file
  values:
    "*":
      KEY_1: value_1
      KEY_2: value_2
```

執行時 `gitlab-ci-local` 會建立暫存檔（內容為 `KEY_1=value_1\nKEY_2=value_2`），並將 `ENV_FILE` 變數設為該檔案路徑，供 job 中的腳本讀取使用。

## 環境變數檔（.env）

在專案根目錄放置 `.env` 檔案，`deploy.sh` 會在執行時載入並 export 其中的變數，供 `envsubst` 展開 Kubernetes 部署模板中的佔位符。
