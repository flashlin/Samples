import { html, LitElement } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import p5 from 'p5';

export interface EditorStatus {
  mode: 'normal' | 'insert' | 'visual';
  cursorX: number;
  cursorY: number;
  cursorVisible: boolean;
}

export interface BufferCell {
  char: string;
  foreground: number[];
  background: number[];
}

@customElement('vim-editor')
export class VimEditor extends LitElement {
  private p5Instance: p5 | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private cursorBlinkInterval: number | null = null;
  private baseCharWidth = 9;
  private lineHeight = 20;
  private textPadding = 2;
  private textOffsetY = 5;
  private statusBarHeight = 24;
  
  @state()
  private cursorVisible = true;
  
  @property({ type: String })
  mode: 'normal' | 'insert' | 'visual' = 'normal';

  @property({ type: Number })
  cursorX = 0;

  @property({ type: Number })
  cursorY = 0;

  @property({ type: Array })
  content: string[] = ['Hello World中文!'];
  
  @state()
  private lastKeyPressed = '';

  private buffer: BufferCell[][] = [];
  private bufferWidth = 0;
  private bufferHeight = 0;

  private getRectY(lineIndex: number): number {
    return this.textPadding + lineIndex * this.lineHeight;
  }

  private getTextY(lineIndex: number): number {
    return this.getRectY(lineIndex) + this.textOffsetY;
  }

  private isFullWidthChar(char: string): boolean {
    if (!char) return false;
    const code = char.charCodeAt(0);
    return code >= 0x4E00 && code <= 0x9FFF || 
           code >= 0x3040 && code <= 0x30FF ||
           code >= 0xAC00 && code <= 0xD7AF ||
           code >= 0xFF00 && code <= 0xFFEF;
  }

  private getCharWidth(char: string): number {
    return this.isFullWidthChar(char) ? this.baseCharWidth * 2 : this.baseCharWidth;
  }

  private getTextXPosition(line: string, charIndex: number): number {
    let x = 60;
    for (let i = 0; i < charIndex && i < line.length; i++) {
      x += this.getCharWidth(line[i]);
    }
    return x;
  }

  getStatus(): EditorStatus {
    return {
      mode: this.mode,
      cursorX: this.cursorX,
      cursorY: this.cursorY,
      cursorVisible: this.cursorVisible,
    };
  }

  getBuffer(): BufferCell[][] {
    return this.buffer;
  }

  setContent(content: string[]) {
    this.content = [...content];
    if (this.buffer.length === 0) {
      this.initializeBuffer();
    }
    this.updateBuffer();
    if (this.p5Instance) {
      this.p5Instance.redraw();
    }
  }

  private initializeBuffer() {
    const editableWidth = Math.floor((800 - 60) / this.baseCharWidth);
    const editableHeight = Math.floor((600 - this.statusBarHeight) / this.lineHeight);
    
    this.bufferWidth = editableWidth;
    this.bufferHeight = editableHeight;
    
    this.buffer = Array(editableHeight).fill(null).map(() =>
      Array(editableWidth).fill(null).map(() => ({
        char: ' ',
        foreground: [255, 255, 255],
        background: [0, 0, 0],
      }))
    );
  }

  updateBuffer() {
    if (this.buffer.length === 0) {
      this.initializeBuffer();
    }
    
    for (let y = 0; y < this.bufferHeight; y++) {
      for (let x = 0; x < this.bufferWidth; x++) {
        const line = this.content[y] || '';
        const char = line[x] || ' ';
        
        const isCursor = y === this.cursorY && x === this.cursorX && this.cursorVisible;
        const isNormalMode = this.mode === 'normal';
        
        this.buffer[y][x] = {
          char,
          foreground: (isCursor && isNormalMode) ? [0, 0, 0] : [255, 255, 255],
          background: (isCursor && isNormalMode) ? [255, 255, 255] : [0, 0, 0],
        };
      }
    }
  }

  firstUpdated() {
    console.log('firstUpdated called');
    
    this.waitForP5AndInitialize();
  }

  private waitForP5AndInitialize() {
    if (typeof (window as any).p5 === 'undefined') {
      console.log('Waiting for p5.js to load...');
      setTimeout(() => this.waitForP5AndInitialize(), 50);
      return;
    }
    
    console.log('p5.js loaded, initializing...');
    this.initializeP5();
  }

  private initializeP5() {
    const width = 800;
    const height = 600;
    
    console.log('Canvas size:', width, height);

    const sketch = (p: p5) => {
      p.setup = () => {
        console.log('p5 setup called');
        const canvas = p.createCanvas(width, height);
        this.canvas = canvas.elt as HTMLCanvasElement;
        
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
        
        this.baseCharWidth = p.textWidth('M');
        
        this.initializeBuffer();
        
        p.noLoop();
      };

      p.draw = () => {
        console.log('Drawing frame...');
        
        this.updateBuffer();
        
        p.background(0);
        
        this.drawBorder(p);
        this.drawEditorBackground(p);
        this.drawLineNumbers(p);
        this.renderBuffer(p);
        this.drawStatusBar(p);
        
        console.log('Frame complete');
      };
    };

    this.p5Instance = new (window as any).p5(sketch, this.shadowRoot as unknown as HTMLElement);

    this.cursorBlinkInterval = window.setInterval(() => {
      this.cursorVisible = !this.cursorVisible;
      if (this.p5Instance) {
        console.log('Triggering redraw...');
        this.p5Instance.redraw();
      }
    }, 500);

    window.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  private handleKeyDown(event: KeyboardEvent) {
    const key = event.key;
    this.lastKeyPressed = key;
    
    event.preventDefault();
    
    if (this.mode === 'normal') {
      this.handleNormalMode(key);
    } else if (this.mode === 'insert') {
      this.handleInsertMode(key);
    }
    
    if (this.p5Instance) {
      this.p5Instance.redraw();
    }
  }

  private handleNormalMode(key: string) {
    switch (key) {
      case 'j':
        this.moveCursorDown();
        break;
      case 'k':
        this.moveCursorUp();
        break;
      case 'h':
        this.moveCursorLeft();
        break;
      case 'l':
        this.moveCursorRight();
        break;
      case '$':
        this.moveCursorToLineEnd();
        break;
      case 'i':
        this.mode = 'insert';
        break;
      case 'a':
        this.moveCursorRight();
        this.mode = 'insert';
        break;
      case 'o':
        this.insertLineBelow();
        break;
    }
  }

  private handleInsertMode(key: string) {
    if (key === 'Escape') {
      this.mode = 'normal';
    } else if (key === 'Backspace') {
      this.handleBackspace();
    } else if (key === 'Enter') {
      this.handleEnter();
    } else if (key.length === 1) {
      this.insertCharacter(key);
    }
  }

  private moveCursorDown() {
    if (this.cursorY < this.content.length - 1) {
      this.cursorY += 1;
      this.adjustCursorX();
    }
  }

  private moveCursorUp() {
    if (this.cursorY > 0) {
      this.cursorY -= 1;
      this.adjustCursorX();
    }
  }

  private moveCursorLeft() {
    if (this.cursorX > 0) {
      this.cursorX -= 1;
    }
  }

  private moveCursorRight() {
    const currentLine = this.content[this.cursorY] || '';
    if (this.cursorX < currentLine.length - 1) {
      this.cursorX += 1;
    }
  }

  private moveCursorToLineEnd() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0) {
      this.cursorX = currentLine.length - 1;
    }
  }

  private adjustCursorX() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0 && this.cursorX >= currentLine.length) {
      this.cursorX = currentLine.length - 1;
    }
  }

  private insertLineBelow() {
    this.content.splice(this.cursorY + 1, 0, '');
    this.cursorY += 1;
    this.cursorX = 0;
    this.mode = 'insert';
  }

  private handleBackspace() {
    if (this.cursorX > 0) {
      const currentLine = this.content[this.cursorY];
      this.content[this.cursorY] = currentLine.substring(0, this.cursorX - 1) + currentLine.substring(this.cursorX);
      this.cursorX -= 1;
    } else if (this.cursorY > 0) {
      const previousLine = this.content[this.cursorY - 1];
      const currentLine = this.content[this.cursorY];
      
      this.cursorX = previousLine.length;
      this.content[this.cursorY - 1] = previousLine + currentLine;
      this.content.splice(this.cursorY, 1);
      this.cursorY -= 1;
    }
  }

  private handleEnter() {
    const currentLine = this.content[this.cursorY];
    const lineBeforeCursor = currentLine.substring(0, this.cursorX);
    const lineAfterCursor = currentLine.substring(this.cursorX);
    
    this.content[this.cursorY] = lineBeforeCursor;
    this.content.splice(this.cursorY + 1, 0, lineAfterCursor);
    
    this.cursorY += 1;
    this.cursorX = 0;
  }

  private insertCharacter(char: string) {
    const currentLine = this.content[this.cursorY];
    this.content[this.cursorY] = 
      currentLine.substring(0, this.cursorX) + 
      char + 
      currentLine.substring(this.cursorX);
    this.cursorX += 1;
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

  private drawBorder(p: p5) {
    p.stroke(100);
    p.noFill();
    p.rect(0, 0, p.width - 1, p.height - 1);
    p.noStroke();
  }

  private drawEditorBackground(p: p5) {
    p.fill(0);
    p.rect(0, 0, p.width, p.height - this.statusBarHeight);
  }

  private drawLineNumbers(p: p5) {
    p.fill(0, 0, 100);
    p.rect(0, 0, 50, p.height - this.statusBarHeight);
    p.fill(255);
    this.content.forEach((_, i) => {
      p.text((i + 1).toString(), 5, this.getTextY(i));
    });
  }

  private renderBuffer(p: p5) {
    for (let y = 0; y < this.bufferHeight && y < this.buffer.length; y++) {
      const line = this.content[y] || '';
      for (let x = 0; x < this.bufferWidth && x < this.buffer[y].length; x++) {
        const cell = this.buffer[y][x];
        const char = line[x] || ' ';
        const charWidth = this.getCharWidth(char);
        const screenX = this.getTextXPosition(line, x);
        
        if (cell.background[0] !== 0 || cell.background[1] !== 0 || cell.background[2] !== 0) {
          p.fill(cell.background[0], cell.background[1], cell.background[2]);
          p.rect(screenX, this.getRectY(y), charWidth, this.lineHeight);
        }
        
        p.fill(cell.foreground[0], cell.foreground[1], cell.foreground[2]);
        p.text(cell.char, screenX, this.getTextY(y));
      }
    }
    
    this.drawInsertCursor(p);
  }

  private drawInsertCursor(p: p5) {
    if (this.mode !== 'insert' || !this.cursorVisible) {
      return;
    }
    
    const line = this.content[this.cursorY] || '';
    const screenX = this.getTextXPosition(line, this.cursorX);
    const screenY = this.getRectY(this.cursorY);
    
    p.stroke(255);
    p.strokeWeight(2);
    p.line(screenX, screenY, screenX, screenY + this.lineHeight);
    p.noStroke();
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