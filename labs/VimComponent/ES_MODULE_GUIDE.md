# ES Module 使用指南 - 程式化建立 p5 實例

## 概述

VimComponent 使用**純 ES module** 的方式來建立 p5.js 實例，完全不依賴全域 `window` 物件。

## 傳統方式 vs ES Module 方式

### ❌ 傳統方式（依賴 window 物件）

```html
<!-- HTML 中載入 p5.js -->
<script src="https://cdn.jsdelivr.net/npm/p5@1.11.10/lib/p5.min.js"></script>

<script>
  // p5 自動掛載到 window.p5
  const myP5 = new p5((p) => {
    p.setup = () => {
      p.createCanvas(400, 400);
    };
    p.draw = () => {
      p.background(220);
    };
  });
</script>
```

**缺點**：
- 污染全域命名空間
- 無法使用 package manager 管理版本
- 不支援 tree-shaking
- TypeScript 類型支援較弱
- 需要額外的 HTTP 請求

---

### ✅ ES Module 方式（程式化）

```typescript
// 使用 ES module import
import p5 from 'p5';

class MyComponent {
  private p5Instance: p5 | null = null;

  initialize() {
    // 直接使用 imported 的 p5 類別
    const sketch = (p: p5) => {
      p.setup = () => {
        p.createCanvas(400, 400);
      };
      
      p.draw = () => {
        p.background(220);
      };
    };

    // 程式化建立實例，不依賴 window
    this.p5Instance = new p5(sketch, document.body);
  }

  cleanup() {
    if (this.p5Instance) {
      this.p5Instance.remove();
      this.p5Instance = null;
    }
  }
}
```

**優點**：
- ✅ 使用 package manager（pnpm/npm）管理版本
- ✅ TypeScript 完整類型支援
- ✅ 支援 tree-shaking（減少 bundle 大小）
- ✅ 不污染全域命名空間
- ✅ 模組化、可維護性高
- ✅ 可以在同一頁面建立多個獨立實例

---

## VimComponent 的實作

### 1. Import p5

```typescript
// vim-editor.ts
import p5 from 'p5';
```

### 2. 定義 p5 實例屬性

```typescript
export class VimEditor extends LitElement {
  private p5Instance: p5 | null = null;
  // ...
}
```

### 3. 建立 sketch 函數

```typescript
private initializeP5() {
  const width = 800;
  const height = 600;
  
  // Sketch 函數接收 p5 實例作為參數
  const sketch = (p: p5) => {
    p.setup = () => {
      const canvas = p.createCanvas(width, height);
      this.canvas = canvas.elt as HTMLCanvasElement;
      
      p.background(0);
      p.fill(255);
      p.textSize(16);
      p.textAlign(p.LEFT, p.TOP);
      p.textFont('monospace');
      
      p.noLoop();
    };

    p.draw = () => {
      this.updateBuffer();
      p.background(0);
      // ... 繪製邏輯
    };
  };

  // ...
}
```

### 4. 程式化建立 p5 實例

```typescript
// 關鍵：直接使用 imported 的 p5 類別
this.p5Instance = new p5(sketch, this.shadowRoot as unknown as HTMLElement);
```

**重點**：
- 第一個參數：sketch 函數
- 第二個參數：容器元素（可以是任何 HTMLElement）
- 在 Web Component 中，使用 `shadowRoot` 來隔離樣式

### 5. 清理實例

```typescript
disconnectedCallback() {
  if (this.p5Instance) {
    this.p5Instance.remove();
  }
  super.disconnectedCallback();
}
```

---

## 完整範例

### 基本範例

```typescript
import p5 from 'p5';

class SimpleP5Component extends HTMLElement {
  private p5Instance: p5 | null = null;

  connectedCallback() {
    const sketch = (p: p5) => {
      p.setup = () => {
        p.createCanvas(400, 400);
      };

      p.draw = () => {
        p.background(220);
        p.fill(255, 0, 0);
        p.ellipse(p.mouseX, p.mouseY, 50, 50);
      };
    };

    this.p5Instance = new p5(sketch, this);
  }

  disconnectedCallback() {
    this.p5Instance?.remove();
  }
}

customElements.define('simple-p5', SimpleP5Component);
```

### 進階範例：多個實例

```typescript
import p5 from 'p5';

class MultiP5Manager {
  private instances: p5[] = [];

  createInstance(container: HTMLElement, color: number) {
    const sketch = (p: p5) => {
      p.setup = () => {
        p.createCanvas(200, 200);
      };

      p.draw = () => {
        p.background(color);
        p.fill(255);
        p.ellipse(100, 100, 50, 50);
      };
    };

    const instance = new p5(sketch, container);
    this.instances.push(instance);
    return instance;
  }

  removeAll() {
    this.instances.forEach(p5 => p5.remove());
    this.instances = [];
  }
}

// 使用
const manager = new MultiP5Manager();
manager.createInstance(document.getElementById('canvas1')!, 200);
manager.createInstance(document.getElementById('canvas2')!, 100);
manager.createInstance(document.getElementById('canvas3')!, 50);
```

---

## 與 Shadow DOM 整合

在 Web Component 中使用 Shadow DOM 可以完全隔離樣式：

```typescript
import { LitElement, html } from 'lit';
import { customElement } from 'lit/decorators.js';
import p5 from 'p5';

@customElement('p5-widget')
export class P5Widget extends LitElement {
  private p5Instance: p5 | null = null;

  firstUpdated() {
    const sketch = (p: p5) => {
      p.setup = () => {
        const canvas = p.createCanvas(400, 400);
        // 設置 canvas 樣式
        canvas.elt.style.display = 'block';
      };

      p.draw = () => {
        p.background(100, 150, 200);
        p.fill(255);
        p.text('Hello from Shadow DOM!', 10, 20);
      };
    };

    // 在 Shadow DOM 中建立實例
    this.p5Instance = new p5(sketch, this.shadowRoot as unknown as HTMLElement);
  }

  disconnectedCallback() {
    this.p5Instance?.remove();
    super.disconnectedCallback();
  }

  render() {
    return html`
      <style>
        :host {
          display: block;
          width: 400px;
          height: 400px;
        }
      </style>
    `;
  }
}
```

---

## TypeScript 類型支援

### 完整類型定義

```typescript
import p5 from 'p5';

// sketch 函數的類型
type SketchFunction = (p: p5) => void;

// 組件類別
class TypedP5Component {
  private p5Instance: p5 | null = null;
  private canvas: HTMLCanvasElement | null = null;

  initialize(container: HTMLElement): void {
    const sketch: SketchFunction = (p: p5) => {
      p.setup = (): void => {
        const canvasObj = p.createCanvas(400, 400);
        this.canvas = canvasObj.elt as HTMLCanvasElement;
      };

      p.draw = (): void => {
        p.background(220);
        // TypeScript 會提供完整的 p5 API 自動完成
        p.fill(p.random(255), p.random(255), p.random(255));
        p.ellipse(p.width / 2, p.height / 2, 100, 100);
      };
    };

    this.p5Instance = new p5(sketch, container);
  }

  redraw(): void {
    this.p5Instance?.redraw();
  }

  cleanup(): void {
    if (this.p5Instance) {
      this.p5Instance.remove();
      this.p5Instance = null;
    }
  }
}
```

---

## 常見模式

### 1. Responsive Canvas

```typescript
const sketch = (p: p5) => {
  p.setup = () => {
    p.createCanvas(p.windowWidth, p.windowHeight);
  };

  p.windowResized = () => {
    p.resizeCanvas(p.windowWidth, p.windowHeight);
  };

  p.draw = () => {
    p.background(220);
    // 繪製邏輯
  };
};

const instance = new p5(sketch, document.body);
```

### 2. noLoop 模式（僅在需要時繪製）

```typescript
const sketch = (p: p5) => {
  p.setup = () => {
    p.createCanvas(400, 400);
    p.noLoop(); // 停止自動重繪
  };

  p.draw = () => {
    p.background(220);
    p.text('This only draws when redraw() is called', 10, 20);
  };
};

const instance = new p5(sketch);

// 手動觸發重繪
button.addEventListener('click', () => {
  instance.redraw();
});
```

### 3. 事件處理

```typescript
const sketch = (p: p5) => {
  p.setup = () => {
    p.createCanvas(400, 400);
  };

  p.draw = () => {
    p.background(220);
  };

  p.mousePressed = () => {
    console.log('Mouse pressed at', p.mouseX, p.mouseY);
    return false; // 防止預設行為
  };

  p.keyPressed = () => {
    if (p.key === 's') {
      p.saveCanvas('myCanvas', 'png');
    }
  };
};

const instance = new p5(sketch);
```

---

## 效能最佳化

### 1. 使用 Instance Mode

```typescript
// ✅ 推薦：Instance mode（可以有多個實例）
const instance = new p5(sketch, container);

// ❌ 避免：Global mode（只能有一個實例）
// function setup() { ... }
// function draw() { ... }
```

### 2. 適當使用 noLoop()

```typescript
const sketch = (p: p5) => {
  p.setup = () => {
    p.createCanvas(800, 600);
    p.noLoop(); // 靜態內容不需要持續重繪
  };

  p.draw = () => {
    // 只在 redraw() 被呼叫時執行
  };
};
```

### 3. 清理資源

```typescript
class P5Component {
  private p5Instance: p5 | null = null;

  destroy() {
    // 清理 p5 實例
    if (this.p5Instance) {
      this.p5Instance.remove(); // 移除 canvas 和事件監聽器
      this.p5Instance = null;
    }
  }
}
```

---

## 總結

使用 ES module 程式化建立 p5 實例的優點：

| 特性 | 傳統方式 | ES Module |
|------|---------|-----------|
| 依賴管理 | 手動 | pnpm/npm |
| TypeScript | 弱支援 | 完整支援 |
| 全域污染 | ❌ 有 | ✅ 無 |
| Tree-shaking | ❌ 不支援 | ✅ 支援 |
| 多實例 | 困難 | ✅ 簡單 |
| 模組化 | 困難 | ✅ 簡單 |
| 現代化 | ❌ | ✅ |

**結論**：ES module 方式是現代 Web 開發的最佳實踐！

