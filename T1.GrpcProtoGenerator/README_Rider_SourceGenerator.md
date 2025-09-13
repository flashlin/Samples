# Rider Source Generator 支援指南

## 問題：Rider 建置時不產生 Source Generator 程式碼

### 解決步驟

#### 1. 確認 Rider 版本
- 需要 **Rider 2021.3+** 才支援 Source Generators
- 建議使用最新版本的 Rider

#### 2. 強制重新建置
```bash
# 在終端中執行
dotnet clean
dotnet build
```

#### 3. Rider IDE 操作
1. **Build → Rebuild Solution**
2. **File → Invalidate Caches and Restart**
3. **View → Tool Windows → Build** 查看建置輸出

#### 4. 檢查 Generated 資料夾
- 生成的檔案在: `DemoServer/Generated/T1.GrpcProtoGenerator/...`
- 如果檔案存在但 IDE 看不到，嘗試右鍵 → **Reload from Disk**

#### 5. 在 Rider 中查看生成檔案
1. 在 **Solution Explorer** 中
2. 檢查 **DemoServer** → **Generated** 資料夾
3. 如果看不到，嘗試 **Show All Files** 選項

#### 6. 確認設定檔案
確保這些檔案已正確設定：
- `DemoServer.csproj` - 包含 `EmitCompilerGeneratedFiles`
- `Directory.Build.props` - Debug 模式設定
- `.editorconfig` - Source Generator 分析器設定

#### 7. 替代方案
如果 Rider 仍然有問題：
```bash
# 使用命令行建置並檢查結果
cd DemoServer
dotnet build -v normal
ls -la Generated/T1.GrpcProtoGenerator/T1.GrpcProtoGenerator.Generators.GrpcWrapperIncrementalGenerator/
```

#### 8. 驗證 Source Generator 運作
建置後應該看到這些檔案：
- `Generated_greet.cs` - 主要生成的服務程式碼
- `Debug_Generator_Loaded.cs` - 確認生成器載入
- `Debug_Generator_Called.cs` - 確認生成器執行

### 注意事項
- Source Generator 在 **Debug** 建置時才會輸出到 `Generated` 資料夾
- Rider 可能需要幾秒鐘才能索引新生成的檔案
- 如果問題持續，嘗試重新啟動 Rider

### 成功指標
✅ 命令行 `dotnet build` 成功
✅ `Generated` 資料夾有檔案
✅ 生成的 `GreeterService` 類別可用
✅ 命名空間為 `DemoServer` (來自 csharp_namespace)