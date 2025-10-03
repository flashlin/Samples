import { Action, ActionPanel, Clipboard, Icon, List, showToast, Toast } from "@raycast/api";
import { useState, useEffect } from "react";
import { exec } from "child_process";
import path from "path";

export default function Command() {
  const [searchText, setSearchText] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const searchDir = "/Users/flash"; // 設定根目錄

  const performSearch = (query: string) => {
    if (!query) {
      setFiles([]);
      return;
    }

    setIsLoading(true);
    
    // Escape single quotes in regex for shell safety
    const escapedRegex = query.replace(/'/g, "'\\''");
    
    // Use full path to fd (Homebrew installation)
    const fdPath = "/opt/homebrew/bin/fd";
    const command = `${fdPath} --type f --no-ignore-vcs '${escapedRegex}' '${searchDir}'`;
    
    exec(command, { maxBuffer: 1024 * 1024 * 10 }, (err, stdout, stderr) => {
      setIsLoading(false);
      
      if (err) {
        console.error("fd error:", err, stderr);
        showToast({ 
          style: Toast.Style.Failure, 
          title: "搜尋失敗", 
          message: stderr || err.message 
        });
        setFiles([]);
        return;
      }
      
      const results = stdout.trim().split("\n").filter(Boolean);
      setFiles(results);
      
      if (results.length === 0) {
        showToast({ 
          style: Toast.Style.Success, 
          title: "無結果", 
          message: `找不到符合 "${query}" 的檔案` 
        });
      }
    });
  };

  useEffect(() => {
    performSearch(searchQuery);
  }, [searchQuery]);

  return (
    <List 
      searchBarPlaceholder="輸入檔案名稱或 pattern，按 Enter 搜尋（例如：\.sql$ 或 test）" 
      onSearchTextChange={setSearchText}
      searchText={searchText}
      isLoading={isLoading}
    >
      {files.length === 0 && !isLoading && searchQuery && (
        <List.EmptyView 
          title="無搜尋結果" 
          description={`找不到符合 "${searchQuery}" 的檔案`}
        />
      )}
      {files.length === 0 && !searchQuery && (
        <List.EmptyView 
          title="輸入搜尋條件" 
          description="輸入檔案名稱或 pattern，按 Enter 開始搜尋"
        />
      )}
      {files.map((file) => (
        <List.Item
          key={file}
          title={path.basename(file)}
          subtitle={file}
          actions={
            <ActionPanel>
              <Action
                title="開始搜尋"
                icon={Icon.MagnifyingGlass}
                onAction={() => {
                  setSearchQuery(searchText);
                }}
              />
              <Action
                title="複製檔案路徑"
                icon={Icon.Clipboard}
                onAction={async () => {
                  await Clipboard.copy(file);
                  showToast({ style: Toast.Style.Success, title: "已複製路徑" });
                }}
              />
              <Action
                title="打開檔案"
                icon={Icon.Document}
                onAction={() => {
                  exec(`open "${file}"`);
                }}
              />
            </ActionPanel>
          }
        />
      ))}
    </List>
  );
}
