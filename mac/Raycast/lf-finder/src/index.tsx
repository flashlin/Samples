import { Action, ActionPanel, Clipboard, Icon, List, showToast, Toast } from "@raycast/api";
import { useState, useEffect } from "react";
import { exec } from "child_process";
import path from "path";

export default function Command() {
  const [regex, setRegex] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const searchDir = "/Users/flash"; // 設定根目錄

  useEffect(() => {
    if (!regex) {
      setFiles([]);
      return;
    }

    setIsLoading(true);
    
    // Escape single quotes in regex for shell safety
    const escapedRegex = regex.replace(/'/g, "'\\''");
    
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
          message: `找不到符合 "${regex}" 的檔案` 
        });
      }
    });
  }, [regex]);

  return (
    <List 
      searchBarPlaceholder="輸入檔案名稱或 pattern（例如：\.sql$ 或 test）" 
      onSearchTextChange={setRegex} 
      isLoading={isLoading}
      throttle
    >
      {files.map((file) => (
        <List.Item
          key={file}
          title={path.basename(file)}
          subtitle={file}
          actions={
            <ActionPanel>
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
