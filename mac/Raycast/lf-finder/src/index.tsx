import { Action, ActionPanel, Clipboard, Icon, List, showToast, Toast } from "@raycast/api";
import { useState, useEffect } from "react";
import { exec } from "child_process";
import path from "path";

export default function Command() {
  const [regex, setRegex] = useState("");
  const [files, setFiles] = useState<string[]>([]);
  const searchDir = "/Users/yourname/Documents"; // 設定根目錄

  useEffect(() => {
    if (!regex) return;

    exec(`fd --regex "${regex}" ${searchDir}`, (err, stdout) => {
      if (err) {
        setFiles([]);
        return;
      }
      const results = stdout.split("\n").filter(Boolean);
      setFiles(results);
    });
  }, [regex]);

  return (
    <List searchBarPlaceholder="輸入 regex 搜尋檔案" onSearchTextChange={setRegex} throttle>
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
