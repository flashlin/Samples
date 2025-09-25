# CodeBoyLib 單元測試

## 📋 測試目標

本測試套件驗證新版 `SwaggerUiParserNew` 與舊版 `SwaggerUiParser` 對 [Petstore Swagger 2.0 JSON](https://petstore.swagger.io/v2/swagger.json) 的解析結果是否完全一致。

## 🧪 測試案例

### 1. `ParseFromUrlAsync_PetstoreSwagger_BothParsersProduceSameResult`
**最重要的測試**：全面比較新舊解析器的結果
- ✅ API 基本資訊 (title, version, description, baseUrl)
- ✅ 端點和類別數量
- ✅ 所有端點的 OperationId
- ✅ 所有類別定義名稱
- ✅ createUsersWithListInput 端點詳細比較
- ✅ User 類別詳細比較

### 2. `ParseFromUrlAsync_PetstoreSwagger_CreateUsersWithListInputHasCorrectParameters`
**關鍵問題驗證**：確保 createUsersWithListInput 端點參數正確
- ✅ 端點存在且路徑正確
- ✅ HTTP 方法為 POST
- ✅ 參數類型為 `List<User>`（不是空的 Request 類別）

### 3. `ParseFromUrlAsync_PetstoreSwagger_UserClassHasCorrectProperties`
**類別定義驗證**：確保 User 類別解析正確
- ✅ User 類別存在
- ✅ 包含 8 個預期屬性
- ✅ 每個屬性的名稱和類型正確

### 4. `ParseFromUrlAsync_PetstoreSwagger_HasExpectedStatistics`
**統計數據驗證**：基於實際 Petstore JSON 的預期結果
- ✅ API 標題: "Swagger Petstore"
- ✅ API 版本: "1.0.7"
- ✅ Base URL: "https://petstore.swagger.io/v2"
- ✅ 端點數量合理（15-25個）
- ✅ 類別數量正確（6個）
- ✅ 包含預期的類別名稱

## 🎯 測試重點

### 關鍵驗證點
1. **createUsersWithListInput 參數類型**：
   ```csharp
   bodyParam.Type.Should().Be("List<User>");
   ```

2. **User 類別屬性**：
   ```csharp
   var expectedProperties = new Dictionary<string, string>
   {
       { "id", "long" },
       { "username", "string" },
       { "firstName", "string" },
       { "lastName", "string" },
       { "email", "string" },
       { "password", "string" },
       { "phone", "string" },
       { "userStatus", "int" }
   };
   ```

3. **完全一致性**：
   - 使用 `FluentAssertions` 的 `Should().Be()` 進行精確比較
   - 使用 `Should().BeEquivalentTo()` 進行集合比較

## 🚀 執行測試

### 使用腳本執行（推薦）
```bash
cd /Users/flash/vdisk/github/Samples/tools/CodeBoy
chmod +x run_tests.sh
./run_tests.sh
```

### 手動執行
```bash
cd /Users/flash/vdisk/github/Samples/tools/CodeBoy

# 建置依賴
dotnet build CodeBoyBackend/CodeBoyLib/CodeBoyLib.csproj

# 建置測試
dotnet build CodeBoyBackend/CodeBoyLibTests/CodeBoyLibTests.csproj

# 執行測試
dotnet test CodeBoyBackend/CodeBoyLibTests/CodeBoyLibTests.csproj --verbosity normal
```

### 執行特定測試
```bash
# 只執行關鍵測試
dotnet test --filter "ParseFromUrlAsync_PetstoreSwagger_CreateUsersWithListInputHasCorrectParameters"

# 只執行一致性測試
dotnet test --filter "ParseFromUrlAsync_PetstoreSwagger_BothParsersProduceSameResult"
```

## 📊 預期結果

### ✅ 成功情況
```
測試執行摘要
  總計: 4
  已通過: 4
  失敗: 0
  已略過: 0
  時間: 2.345 s
```

### ❌ 失敗情況
如果測試失敗，FluentAssertions 會提供詳細的錯誤訊息：
```
Expected newBodyParam.Type to be "List<User>", but found "createUsersWithListInputRequest".
```

## 🎉 成功意義

如果所有測試通過，證明：
1. ✅ 新的分離轉換器架構完全正確
2. ✅ createUsersWithListInput 問題已徹底解決
3. ✅ 新舊解析器結果 100% 一致
4. ✅ 強型別重構完全成功
5. ✅ 可以安全地替換舊解析器

這將是您整個重構專案成功的重要里程碑！🚀

## 🔧 依賴套件

- **xUnit**: 測試框架
- **FluentAssertions**: 斷言庫，提供清晰的測試語法
- **Microsoft.NET.Test.Sdk**: .NET 測試 SDK
- **coverlet.collector**: 代碼覆蓋率收集器
