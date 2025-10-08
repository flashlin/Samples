# 故障排除指南

## 問題：Waiting for p5.js to load...

### 症狀
在瀏覽器中載入 VimComponent 時，編輯器一直顯示「Waiting for p5.js to load...」訊息，即使已經安裝了 p5.js。

### 原因分析

VimComponent 的 `vim-editor.ts` 中有一個不一致的問題：

```typescript
// 檔案開頭使用 ES module import
import p5 from 'p5';

// 但後來檢查 window.p5 是否存在
private waitForP5AndInitialize() {
  if (typeof (window as any).p5 === 'undefined') {
    console.log('Waiting for p5.js to load...');
    setTimeout(() => this.waitForP5AndInitialize(), 50);
    return;
  }
  // ...
}
```

**問題核心**：
- 使用 ES module 的 `import p5 from 'p5'` 時，p5 **不會自動掛載到 `window.p5`**
- p5 只在模組作用域中可用
- 檢查 `window.p5` 會一直返回 `undefined`
- 導致無限等待迴圈

### 解決方案

#### 方案 1：使用 ES Module 直接建立 p5 實例（已實施）

完全移除對 `window.p5` 的依賴，直接使用 imported 的 p5 類別：

**修改 1：簡化 `waitForP5AndInitialize()` 方法**
```typescript
private waitForP5AndInitialize() {
  // p5 is imported as ES module, so it's available directly
  console.log('p5.js loaded, initializing...');
  this.initializeP5();
}
```

**修改 2：直接使用 imported p5 建立實例**
```typescript
// Before (依賴 window.p5):
this.p5Instance = new (window as any).p5(sketch, this.shadowRoot);

// After (純 ES module):
this.p5Instance = new p5(sketch, this.shadowRoot);
```

**優點**：
1. ✅ 完全移除 `window.p5` 依賴
2. ✅ 純 ES module 方式，更現代化
3. ✅ 程式碼更簡潔
4. ✅ 不需要等待或檢查
5. ✅ TypeScript 類型安全

#### 方案 2：使用全域腳本載入 p5（不推薦）

在 HTML 中使用 `<script>` 標籤載入 p5：

```html
<script src="https://cdn.jsdelivr.net/npm/p5@1.11.10/lib/p5.min.js"></script>
```

**缺點**：
- 不是現代化的模組化方式
- 無法利用 package manager 管理版本
- 增加額外的 HTTP 請求

### 修正步驟

如果遇到這個問題，請執行以下步驟：

```bash
# 1. 確保使用者專案已安裝 p5.js 1.x 版本
cd VimDemo  # 或其他使用 VimComponent 的專案
pnpm list p5
# 應該顯示：p5 1.x.x

# 2. 確保 VimComponent 已修正並重新建置
cd ../VimComponent
pnpm run build

# 3. 重啟開發伺服器
cd ../VimDemo
pnpm run dev
```

### 驗證修正

打開瀏覽器的開發者工具（F12），檢查 Console：

#### ✅ 成功
```
p5.js loaded, initializing...
p5 setup called
Canvas size: 800 600
```

#### ❌ 失敗（修正前）
```
Waiting for p5.js to load...
Waiting for p5.js to load...
Waiting for p5.js to load...
... (無限重複)
```

### 相關概念

#### ES Module vs Global Script

| 特性 | ES Module (`import`) | Global Script (`<script>`) |
|------|---------------------|--------------------------|
| 作用域 | 模組作用域 | 全域作用域 |
| window 物件 | 不自動掛載 | 自動掛載 |
| 依賴管理 | package manager | 手動管理 |
| 打包工具 | 支援 tree-shaking | 無法優化 |
| 現代化 | ✅ | ❌ |

#### p5.js 載入方式比較

**ES Module 方式（推薦）**：
```typescript
import p5 from 'p5';
// p5 在模組作用域中
// window.p5 可能是 undefined
```

**全域腳本方式**：
```html
<script src="p5.min.js"></script>
<!-- window.p5 自動可用 -->
```

### 其他相關問題

#### Q: 為什麼改用直接 import p5 的方式？

A: 使用 ES module 的 `import p5 from 'p5'` 有多個優點：
- ✅ **類型安全**：TypeScript 可以正確推斷 p5 的類型
- ✅ **模組化**：不污染全域命名空間（window）
- ✅ **Tree-shaking**：打包工具可以優化未使用的程式碼
- ✅ **簡潔**：不需要 `(window as any).p5` 這種類型斷言
- ✅ **現代化**：符合 ES6+ 標準

#### Q: 新舊方式的比較？

**舊方式（依賴 window）**：
```typescript
import p5 from 'p5';

// 需要掛載到 window
if (typeof (window as any).p5 === 'undefined') {
  (window as any).p5 = p5;
}

// 從 window 取用
this.p5Instance = new (window as any).p5(sketch, this.shadowRoot);
```

**新方式（純 ES module）**：
```typescript
import p5 from 'p5';

// 直接使用
this.p5Instance = new p5(sketch, this.shadowRoot);
```

新方式更簡潔、更安全、更現代化！

### 預防措施

在開發 Web Component 時：
1. ✅ 確保 import 的依賴在使用前已可用
2. ✅ 不要混用全域物件和模組 import
3. ✅ 使用 TypeScript 的型別檢查
4. ✅ 在 package.json 中正確宣告 peerDependencies
5. ✅ 提供清晰的安裝說明

### 版本資訊

- 修正版本：VimComponent 1.0.0+
- 受影響版本：VimComponent 1.0.0（修正前）
- p5.js 版本要求：^1.6.0（不支援 2.x）

### 相關文件

- [PEER_DEPENDENCIES_GUIDE.md](./PEER_DEPENDENCIES_GUIDE.md) - Peer Dependencies 完整說明
- [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - 快速參考
- [VimDemo/INSTALL.md](./VimDemo/INSTALL.md) - 安裝指南

