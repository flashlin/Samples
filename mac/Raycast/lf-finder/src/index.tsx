import { Action, ActionPanel, Clipboard, Icon, List, showToast, Toast } from "@raycast/api";
import { useState, useEffect } from "react";
import React from "react";
import fs from "fs";
import path from "path";
import { exec } from "child_process";

export default function Command() {
  const [regex, setRegex] = useState("");
  const [files, setFiles] = useState<string[]>([]);

  useEffect(() => {
    if (!regex) return;

    try {
      const pattern = new RegExp(regex, "i");
      const searchDir = "/Users/yourname"; // 你要搜尋的根目錄

      const matched: string[] = [];
      function walk(dir: string) {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            if (!entry.name.startsWith(".")) walk(fullPath); // 避免 .git node_modules
          } else if (pattern.test(entry.name)) {
            matched.push(fullPath);
          }
        }
      }

      walk(searchDir);
      setFiles(matched);
    } catch (err) {
      setFiles([]);
    }
  }, [regex]);

  return (
    <List
      searchBarPlaceholder="輸入 regex 搜尋檔案"
      onSearchTextChange={setRegex}
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

