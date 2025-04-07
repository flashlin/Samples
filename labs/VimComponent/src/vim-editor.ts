import { html, LitElement } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import p5 from 'p5';

@customElement('vim-editor')
export class VimEditor extends LitElement {
  private p5Instance: p5 | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private cursorBlinkInterval: number | null = null;
  
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
  }

  disconnectedCallback() {
    if (this.cursorBlinkInterval) {
      clearInterval(this.cursorBlinkInterval);
    }
    if (this.p5Instance) {
      this.p5Instance.remove();
    }
    super.disconnectedCallback();
  }

  private drawLineNumbers(p: p5) {
    p.fill(0, 0, 100);
    p.rect(0, 0, 50, p.height-30);
    p.fill(255);
    this.content.forEach((_, i) => {
      const y = 20 + i * 20;
      p.text((i + 1).toString(), 5, y);
    });
  }

  private drawContent(p: p5) {
    p.fill(255);
    this.content.forEach((line, i) => {
      const y = 20 + i * 20;
      p.text(line, 60, y);
    });
  }

  private drawCursor(p: p5) {
    if (this.cursorVisible) {
      p.fill(255);
      const cursorX = 60 + p.textWidth(this.content[this.cursorY].substring(0, this.cursorX));
      const cursorY = 20 + this.cursorY * 20;
      p.rect(cursorX, cursorY - 16, 2, 20);
    }
  }

  private drawStatusBar(p: p5) {
    p.fill(0, 0, 100);
    p.rect(0, p.height - 30, p.width, 30);
    p.fill(255);
    p.text(`Mode: ${this.mode} | Line: ${this.cursorY + 1}, Col: ${this.cursorX + 1}`, 10, p.height - 10);
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