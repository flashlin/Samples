import { html, LitElement } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import p5 from 'p5';

@customElement('vim-editor')
export class VimEditor extends LitElement {
  private p5Instance: p5 | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private cursorBlinkInterval: number | null = null;
  private charWidth = 9; // 假設等寬字體的字符寬度為 9 像素
  private lineHeight = 20; // 行高
  private baseLine = 14; // 文字基準線
  private textPadding = 2; // 文字上下間距
  private statusBarHeight = 24; // 狀態列高度
  
  @state()
  private cursorVisible = true;
  
  @property({ type: String })
  mode: 'normal' | 'insert' | 'visual' = 'normal';

  @property({ type: Number })
  cursorX = 0;

  @property({ type: Number })
  cursorY = 0;

  @property({ type: Array })
  content: string[] = ['Hello World!'];
  
  @state()
  private lastKeyPressed = '';

  // 計算文字的 Y 座標
  private getTextY(lineIndex: number): number {
    return this.textPadding + lineIndex * this.lineHeight + this.baseLine;
  }

  // 計算游標的 Y 座標
  private getCursorY(lineIndex: number): number {
    return this.textPadding + lineIndex * this.lineHeight + this.baseLine;
  }

  firstUpdated() {
    console.log('firstUpdated called');
    
    // 使用固定大小而不是容器大小
    const width = 800;
    const height = 600;
    
    console.log('Canvas size:', width, height);

    const sketch = (p: p5) => {
      p.setup = () => {
        console.log('p5 setup called');
        const canvas = p.createCanvas(width, height);
        this.canvas = canvas.elt as HTMLCanvasElement;
        
        // 設置 canvas 樣式
        canvas.elt.style.cssText = `
          display: block !important;
          width: 800px !important;
          height: 600px !important;
          position: absolute !important;
          left: 0 !important;
          top: 0 !important;
        `;
        
        p.background(0);
        p.fill(255);
        p.textSize(16);
        p.textAlign(p.LEFT, p.TOP);
        p.textFont('monospace');
        
        // 計算字符寬度
        this.charWidth = p.textWidth('M');
        
        // 關閉自動循環
        p.noLoop();
      };

      p.draw = () => {
        console.log('Drawing frame...');
        // 清除背景
        p.background(0);
        
        // 繪製邊框
        p.stroke(100);
        p.noFill();
        p.rect(0, 0, p.width-1, p.height-1);
        p.noStroke();
        
        // 繪製編輯區域背景
        p.fill(0);
        p.rect(0, 0, p.width, p.height - this.statusBarHeight);
        
        console.log('Drawing line numbers...');
        this.drawLineNumbers(p);
        
        console.log('Drawing content...');
        this.drawContent(p);
        
        console.log('Drawing cursor...');
        this.drawCursor(p);
        
        console.log('Drawing status bar...');
        this.drawStatusBar(p);
        
        console.log('Frame complete');
      };
    };

    // 使用 window.p5 而不是直接導入的 p5
    this.p5Instance = new (window as any).p5(sketch, this.shadowRoot as unknown as HTMLElement);

    // 設置游標閃爍
    this.cursorBlinkInterval = window.setInterval(() => {
      this.cursorVisible = !this.cursorVisible;
      // 直接調用 p5 的 redraw
      if (this.p5Instance) {
        console.log('Triggering redraw...');
        this.p5Instance.redraw();
      }
    }, 500);

    // 添加鍵盤事件監聽器
    window.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  private handleKeyDown(event: KeyboardEvent) {
    const key = event.key;
    this.lastKeyPressed = key;
    
    // 避免瀏覽器默認行為（例如頁面滾動）
    event.preventDefault();
    
    // 處理 Vim 移動按鍵
    if (this.mode === 'normal') {
      switch (key) {
        case 'j': // 向下移動
          if (this.cursorY < this.content.length - 1) {
            this.cursorY += 1;
          }
          break;
        case 'k': // 向上移動
          if (this.cursorY > 0) {
            this.cursorY -= 1;
          }
          break;
        case 'h': // 向左移動
          if (this.cursorX > 0) {
            this.cursorX -= 1;
          }
          break;
        case 'l': // 向右移動
          if (this.content[this.cursorY] && this.cursorX < this.content[this.cursorY].length - 1) {
            this.cursorX += 1;
          }
          break;
        case 'i': // 進入插入模式
          this.mode = 'insert';
          break;
        case 'a': // 往右移動後進入插入模式
          // 如果游標不在行尾，則向右移動一格
          if (this.content[this.cursorY] && this.cursorX < this.content[this.cursorY].length - 1) {
            this.cursorX += 1;
          }
          this.mode = 'insert';
          break;
      }
    } else if (this.mode === 'insert') {
      // 在插入模式中，ESC 鍵返回正常模式
      if (key === 'Escape') {
        this.mode = 'normal';
      } else if (key === 'Backspace') {
        // 刪除游標前的字符
        if (this.cursorX > 0) {
          // 如果游標不在行首，刪除前一個字符
          const currentLine = this.content[this.cursorY];
          this.content[this.cursorY] = currentLine.substring(0, this.cursorX - 1) + currentLine.substring(this.cursorX);
          this.cursorX -= 1;
        } else if (this.cursorY > 0) {
          // 如果游標在行首但不是第一行，將該行合併到上一行
          const previousLine = this.content[this.cursorY - 1];
          const currentLine = this.content[this.cursorY];
          
          // 設置新的游標位置
          this.cursorX = previousLine.length;
          
          // 合併兩行
          this.content[this.cursorY - 1] = previousLine + currentLine;
          
          // 刪除當前行
          this.content.splice(this.cursorY, 1);
          
          // 移動游標到上一行
          this.cursorY -= 1;
        }
      } else if (key === 'Enter') {
        // 處理回車鍵，插入新行
        const currentLine = this.content[this.cursorY];
        
        // 將當前行分割成兩部分
        const lineBeforeCursor = currentLine.substring(0, this.cursorX);
        const lineAfterCursor = currentLine.substring(this.cursorX);
        
        // 更新當前行
        this.content[this.cursorY] = lineBeforeCursor;
        
        // 在當前行之後插入新行
        this.content.splice(this.cursorY + 1, 0, lineAfterCursor);
        
        // 移動游標到下一行的開頭
        this.cursorY += 1;
        this.cursorX = 0;
      } else if (key.length === 1) {
        // 處理普通字符輸入
        const currentLine = this.content[this.cursorY];
        
        // 在游標位置插入字符
        this.content[this.cursorY] = 
          currentLine.substring(0, this.cursorX) + 
          key + 
          currentLine.substring(this.cursorX);
        
        // 移動游標位置
        this.cursorX += 1;
      }
    }
    
    // 重新繪製以更新游標位置和顯示按下的按鍵
    if (this.p5Instance) {
      this.p5Instance.redraw();
    }
  }

  disconnectedCallback() {
    if (this.cursorBlinkInterval) {
      clearInterval(this.cursorBlinkInterval);
    }
    if (this.p5Instance) {
      this.p5Instance.remove();
    }
    // 移除鍵盤事件監聽器
    window.removeEventListener('keydown', this.handleKeyDown.bind(this));
    super.disconnectedCallback();
  }

  private drawLineNumbers(p: p5) {
    p.fill(0, 0, 100);
    p.rect(0, 0, 50, p.height - this.statusBarHeight);
    p.fill(255);
    this.content.forEach((_, i) => {
      p.text((i + 1).toString(), 5, this.getTextY(i));
    });
  }

  private drawContent(p: p5) {
    p.fill(255);
    this.content.forEach((line, i) => {
      p.text(line, 60, this.getTextY(i));
    });
  }

  private drawCursor(p: p5) {
    if (this.cursorVisible) {
      // 使用反色效果
      p.push();
      p.fill(255);
      // 根據當前游標位置計算 X 座標
      const cursorX = 60 + this.cursorX * this.charWidth;
      
      // 使用 getCursorY 方法來計算游標位置
      const cursorY = this.getCursorY(this.cursorY);
      
      // 繪製一個完整字符大小的方塊
      p.rect(cursorX, cursorY, this.charWidth, this.lineHeight);
      
      // 在方塊上用黑色繪製字符
      p.fill(0);
      if (this.content[this.cursorY] && this.content[this.cursorY][this.cursorX]) {
        p.text(this.content[this.cursorY][this.cursorX], cursorX, this.getTextY(this.cursorY));
      }
      p.pop();
    }
  }

  private drawStatusBar(p: p5) {
    // 計算編輯區域的高度
    const editorHeight = p.height - this.statusBarHeight;
    // 將狀態列放在編輯區域的底部
    const statusY = editorHeight;
    
    // 繪製狀態列背景 - 使用暗灰色而不是藍色
    p.fill(50); // 暗灰色背景
    p.rect(0, statusY, p.width, this.statusBarHeight);
    
    // 繪製狀態列文字，包含最後按下的按鍵
    p.fill(255);
    let statusText = `Mode: ${this.mode} | Line: ${this.cursorY + 1}, Col: ${this.cursorX + 1}`;
    if (this.lastKeyPressed) {
      statusText += ` | Key: "${this.lastKeyPressed}"`;
    }
    
    p.text(
      statusText,
      10,
      statusY + 3 // 計算垂直居中位置
    );
  }

  render() {
    return html`
      <style>
        :host {
          display: block;
          width: 800px;
          height: 600px;
          background-color: #000;
          position: relative;
        }
      </style>
    `;
  }
} 