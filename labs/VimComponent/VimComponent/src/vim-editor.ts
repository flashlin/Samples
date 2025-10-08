import { html, LitElement } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import p5 from 'p5';
import exampleText from './example.txt?raw';
import { EditorMode, EditorStatus, BufferCell, EditorModeHandler } from './vimEditorTypes';
import { ModeHandlerRegistry } from './handlers';

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
  private hiddenInput: HTMLInputElement | null = null;
  
  @state()
  private cursorVisible = true;
  
  private _mode: EditorMode = EditorMode.Normal;
  
  @property({ type: String })
  get mode(): EditorMode {
    return this._mode;
  }
  
  set mode(newMode: EditorMode) {
    if (this._mode === newMode) return;
    
    const oldMode = this._mode;
    this._mode = newMode;
    
    if (this.modeHandlerRegistry) {
      const oldHandler = this.modeHandlerRegistry.getHandler(oldMode);
      const newHandler = this.modeHandlerRegistry.getHandler(newMode);
      
      oldHandler.onExit(this);
      this.previousModeHandler = this.currentModeHandler;
      this.currentModeHandler = newHandler;
      newHandler.onEnter(this);
    }
    
    this.requestUpdate('mode', oldMode);
  }

  @property({ type: Number })
  cursorX = 0;

  @property({ type: Number })
  cursorY = 0;

  @property({ type: Array })
  content: string[] = exampleText.split('\n');
  
  @state()
  private lastKeyPressed = '';

  private buffer: BufferCell[][] = [];
  private bufferWidth = 0;
  private bufferHeight = 0;
  private isComposing = false;
  
  private visualStartX = 0;
  private visualStartY = 0;
  
  private numberPrefix = '';
  
  private scrollOffsetX = 0;
  private scrollOffsetY = 0;

  private fastJumpMatches: Array<{ x: number; y: number; label: string }> = [];
  private fastJumpInput = '';
  private previousMode: EditorMode.Normal | EditorMode.Visual | EditorMode.VisualLine = EditorMode.Normal;
  
  private keyBuffer = '';
  private visualKeyBuffer = '';
  
  private searchKeyword = '';
  private searchMatches: Array<{ y: number; x: number }> = [];
  private currentMatchIndex = -1;
  private searchHistory: Array<{ keyword: string; matches: Array<{ y: number; x: number }> }> = [];
  
  private tMarks: Array<{ y: number; x: number }> = [];
  
  private modeHandlerRegistry!: ModeHandlerRegistry;
  private currentModeHandler!: EditorModeHandler;
  private previousModeHandler!: EditorModeHandler;
  
  private commandPatterns = [
    { pattern: 'gg', action: () => { this.moveToFirstLine(); } },
    { pattern: 'diw', action: () => { this.saveHistory(); this.deleteInnerWord(); } },
    { pattern: 'di%', action: () => { this.saveHistory(); this.deleteInnerBracket(); } },
    { pattern: 'di`', action: () => { this.saveHistory(); this.deleteInnerQuote('`'); } },
    { pattern: "di'", action: () => { this.saveHistory(); this.deleteInnerQuote("'"); } },
    { pattern: 'di"', action: () => { this.saveHistory(); this.deleteInnerQuote('"'); } },
    { pattern: 'da(', action: () => { this.saveHistory(); this.deleteAroundBracket('(', ')'); } },
    { pattern: 'da)', action: () => { this.saveHistory(); this.deleteAroundBracket('(', ')'); } },
    { pattern: 'da[', action: () => { this.saveHistory(); this.deleteAroundBracket('[', ']'); } },
    { pattern: 'da]', action: () => { this.saveHistory(); this.deleteAroundBracket('[', ']'); } },
    { pattern: 'da{', action: () => { this.saveHistory(); this.deleteAroundBracket('{', '}'); } },
    { pattern: 'da}', action: () => { this.saveHistory(); this.deleteAroundBracket('{', '}'); } },
    { pattern: 'da<', action: () => { this.saveHistory(); this.deleteAroundBracket('<', '>'); } },
    { pattern: 'da>', action: () => { this.saveHistory(); this.deleteAroundBracket('<', '>'); } },
    { pattern: 'da`', action: () => { this.saveHistory(); this.deleteAroundQuote('`'); } },
    { pattern: "da'", action: () => { this.saveHistory(); this.deleteAroundQuote("'"); } },
    { pattern: 'da"', action: () => { this.saveHistory(); this.deleteAroundQuote('"'); } },
    { pattern: 'da%', action: () => { this.saveHistory(); this.deleteAroundAnyBracket(); } },
    { pattern: 'dw', action: () => { this.saveHistory(); this.deleteWord(); } },
    { pattern: 'de', action: () => { this.saveHistory(); this.deleteToWordEnd(); } },
    { pattern: 'i', action: () => { this.enterInsertMode(); } },
    { pattern: 't', action: () => { 
      const normalHandler = this.modeHandlerRegistry.getHandler(EditorMode.Normal) as any;
      normalHandler.addTMark(this);
    } },
    { pattern: 'T', action: () => { 
      const normalHandler = this.modeHandlerRegistry.getHandler(EditorMode.Normal) as any;
      normalHandler.clearTMarks(this);
    } },
    { pattern: 'a', action: () => { 
      const currentLine = this.content[this.cursorY] || '';
      if (this.cursorX < currentLine.length) {
        this.cursorX += 1;
      }
      this.mode = EditorMode.Insert;
      this.updateInputPosition();
      this.hiddenInput?.focus();
    } },
    { pattern: 'o', action: () => { this.insertLineBelow(); this.hiddenInput?.focus(); } },
    { pattern: 'p', action: () => { this.saveHistory(); this.pasteAfterCursor(); } },
    { pattern: 'v', action: () => { 
      this.mode = EditorMode.Visual;
      this.visualStartX = this.cursorX;
      this.visualStartY = this.cursorY;
    } },
    { pattern: 'V', action: () => { 
      this.mode = EditorMode.VisualLine;
      this.visualStartX = this.cursorX;
      this.visualStartY = this.cursorY;
    } },
    { pattern: 'f', action: () => { 
      this.previousMode = EditorMode.Normal;
      this.mode = EditorMode.FastJump;
      this.fastJumpMatches = [];
      this.fastJumpInput = '';
    } },
    { pattern: 'u', action: () => { this.undo(); } },
    { pattern: '%', action: () => { this.jumpToMatchingBracket(); } },
  ];

  private history: Array<{ content: string[]; cursorX: number; cursorY: number }> = [];
  private historyIndex = -1;
  private maxHistorySize = 100;

  private getRectY(lineIndex: number): number {
    return this.textPadding + lineIndex * this.lineHeight;
  }

  private getTextY(lineIndex: number): number {
    return this.getRectY(lineIndex) + this.textOffsetY;
  }

  private generateLabel(index: number): string {
    const letters = 'abcdefghijklmnopqrstuvwxyz';
    let label = '';
    let current = index;
    
    do {
      label = letters[current % 26] + label;
      current = Math.floor(current / 26) - 1;
    } while (current >= 0);
    
    return label;
  }

  private processKeyBuffer(): boolean {
    const sortedPatterns = [...this.commandPatterns].sort((a, b) => b.pattern.length - a.pattern.length);
    
    for (const { pattern, action } of sortedPatterns) {
      if (this.keyBuffer === pattern) {
        this.keyBuffer = '';
        action();
        return true;
      }
    }
    
    const dNumberJMatch = /^d(\d+)j$/.exec(this.keyBuffer);
    if (dNumberJMatch) {
      const count = parseInt(dNumberJMatch[1], 10);
      this.keyBuffer = '';
      this.saveHistory();
      this.deleteLinesDown(count);
      return true;
    }
    
    const dNumberKMatch = /^d(\d+)k$/.exec(this.keyBuffer);
    if (dNumberKMatch) {
      const count = parseInt(dNumberKMatch[1], 10);
      this.keyBuffer = '';
      this.saveHistory();
      this.deleteLinesUp(count);
      return true;
    }
    
    const hasPartialMatch = sortedPatterns.some(({ pattern }) => 
      pattern.startsWith(this.keyBuffer)
    );
    
    const hasNumberPattern = /^d\d*[jk]?$/.test(this.keyBuffer);
    
    if (!hasPartialMatch && !hasNumberPattern) {
      this.keyBuffer = '';
      return false;
    }
    
    return false;
  }

  private findMatchesInVisibleRange(targetChar: string): Array<{ x: number; y: number; label: string }> {
    const matches: Array<{ x: number; y: number; label: string }> = [];
    let matchIndex = 0;
    
    for (let bufferY = 0; bufferY < this.bufferHeight; bufferY++) {
      const contentY = bufferY + this.scrollOffsetY;
      if (contentY >= this.content.length) {
        break;
      }
      
      const line = this.content[contentY] || '';
      
      for (let contentX = 0; contentX < line.length; contentX++) {
        if (line[contentX] === targetChar) {
          matches.push({
            x: contentX,
            y: contentY,
            label: this.generateLabel(matchIndex)
          });
          matchIndex++;
        }
      }
    }
    
    return matches;
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
      searchKeyword: this.searchKeyword.length > 0 ? this.searchKeyword : undefined,
      searchMatchCount: this.searchMatches.length > 0 ? this.searchMatches.length : undefined,
    };
  }

  getBuffer(): BufferCell[][] {
    return this.buffer;
  }

  getScrollOffset(): { x: number; y: number } {
    return {
      x: this.scrollOffsetX,
      y: this.scrollOffsetY,
    };
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

  load(text: string): void {
    const lines = text.split('\n');
    this.setContent(lines);
    this.cursorX = 0;
    this.cursorY = 0;
    this.scrollOffsetX = 0;
    this.scrollOffsetY = 0;
    this.mode = EditorMode.Normal;
  }

  getDisplayColumn(): number {
    const currentLine = this.content[this.cursorY] || '';
    let displayCol = 0;
    
    for (let i = 0; i < this.cursorX && i < currentLine.length; i++) {
      displayCol += this.isFullWidthChar(currentLine[i]) ? 2 : 1;
    }
    
    return displayCol;
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
    
    for (let bufferY = 0; bufferY < this.bufferHeight; bufferY++) {
      for (let bufferX = 0; bufferX < this.bufferWidth; bufferX++) {
        const contentY = bufferY + this.scrollOffsetY;
        const contentX = bufferX + this.scrollOffsetX;
        
        const line = this.content[contentY] || '';
        const char = line[contentX] || ' ';
        
        const isCursor = contentY === this.cursorY && contentX === this.cursorX && this.cursorVisible;
        const isNormalMode = this.mode === EditorMode.Normal;
        const isSearchMode = this.mode === EditorMode.FastSearch;
        
        const isVisualSelection = this.mode === EditorMode.Visual && this.isInVisualSelection(contentY, contentX);
        const isVisualLineSelection = this.mode === EditorMode.VisualLine && this.isInVisualLineSelection(contentY);
        
        const isFastJumpMatch = this.mode === EditorMode.FastMatch && 
          this.fastJumpMatches.some(match => match.x === contentX && match.y === contentY);
        
        const isSearchMatch = (this.mode === EditorMode.FastSearch || this.mode === EditorMode.MultiInsert) && 
          this.isInSearchMatch(contentY, contentX);
        
        const isCurrentSearchMatch = (this.mode === EditorMode.FastSearch || this.mode === EditorMode.MultiInsert) && 
          this.currentMatchIndex >= 0 && 
          this.isInSearchMatch(contentY, contentX, this.currentMatchIndex);
        
        const isHighlighted = isVisualSelection || isVisualLineSelection || isFastJumpMatch || isSearchMatch;
        
        this.buffer[bufferY][bufferX] = {
          char,
          foreground: (isCursor && (isNormalMode || isSearchMode)) ? [0, 0, 0] : isCurrentSearchMatch ? [0, 0, 0] : isHighlighted ? [0, 0, 0] : [255, 255, 255],
          background: (isCursor && (isNormalMode || isSearchMode)) ? [255, 255, 255] : isCurrentSearchMatch ? [255, 165, 0] : isHighlighted ? [100, 149, 237] : [0, 0, 0],
        };
      }
    }
  }

  private isInVisualSelection(y: number, x: number): boolean {
    const startY = Math.min(this.visualStartY, this.cursorY);
    const endY = Math.max(this.visualStartY, this.cursorY);
    
    if (y < startY || y > endY) {
      return false;
    }
    
    if (startY === endY) {
      const startX = Math.min(this.visualStartX, this.cursorX);
      const endX = Math.max(this.visualStartX, this.cursorX);
      return x >= startX && x <= endX;
    }
    
    if (y === startY) {
      const startX = this.visualStartY === startY ? this.visualStartX : this.cursorX;
      return x >= startX;
    }
    
    if (y === endY) {
      const endX = this.visualStartY === endY ? this.visualStartX : this.cursorX;
      return x <= endX;
    }
    
    return true;
  }

  private isInVisualLineSelection(y: number): boolean {
    const startY = Math.min(this.visualStartY, this.cursorY);
    const endY = Math.max(this.visualStartY, this.cursorY);
    return y >= startY && y <= endY;
  }

  private isInSearchMatch(y: number, x: number, matchIndex?: number): boolean {
    if (this.searchKeyword.length === 0 || this.searchMatches.length === 0) {
      return false;
    }
    
    if (matchIndex !== undefined) {
      const match = this.searchMatches[matchIndex];
      if (!match) return false;
      return y === match.y && x >= match.x && x < match.x + this.searchKeyword.length;
    }
    
    for (const match of this.searchMatches) {
      if (y === match.y && x >= match.x && x < match.x + this.searchKeyword.length) {
        return true;
      }
    }
    
    return false;
  }

  firstUpdated() {
    this.modeHandlerRegistry = new ModeHandlerRegistry();
    this.currentModeHandler = this.modeHandlerRegistry.getHandler(EditorMode.Normal);
    this.previousModeHandler = this.currentModeHandler;
    
    this.createHiddenInput();
    this.waitForP5AndInitialize();
  }

  private createHiddenInput() {
    this.hiddenInput = document.createElement('input');
    this.hiddenInput.setAttribute('type', 'text');
    this.hiddenInput.style.cssText = `
      position: absolute;
      width: 200px;
      height: 20px;
      border: 1px solid rgba(100, 100, 100, 0.3);
      outline: none;
      background: rgba(0, 0, 0, 0.2);
      color: rgb(255, 255, 255);
      caret-color: rgb(255, 255, 255);
      font-size: 16px;
      font-family: monospace;
      padding: 2px;
      margin: 0;
      z-index: 1000;
      pointer-events: none;
      border-radius: 2px;
      opacity: 0;
      transition: opacity 0.1s ease;
    `;
    this.shadowRoot?.appendChild(this.hiddenInput);
    
    this.hiddenInput.addEventListener('compositionstart', (e) => {
      console.log('Composition start:', e.data);
      this.isComposing = true;
      if (this.hiddenInput) {
        this.hiddenInput.style.opacity = '1';
      }
    });
    
    this.hiddenInput.addEventListener('compositionend', (e) => {
      console.log('Composition end:', e.data);
      this.isComposing = false;
      if (this.hiddenInput) {
        this.hiddenInput.style.opacity = '0';
      }
      if (this.mode === EditorMode.Insert && e.data) {
        for (const char of e.data) {
          this.insertCharacter(char);
        }
        if (this.p5Instance) {
          this.p5Instance.redraw();
        }
      } else if (this.mode === EditorMode.TInsert && e.data) {
        const handler = this.currentModeHandler as any;
        for (const char of e.data) {
          if (handler.tInsertCharacter) {
            handler.tInsertCharacter(this, char);
          }
        }
        if (this.p5Instance) {
          this.p5Instance.redraw();
        }
      } else if (this.mode === EditorMode.MultiInsert && e.data) {
        const handler = this.currentModeHandler as any;
        for (const char of e.data) {
          if (handler.multiInsertCharacter) {
            handler.multiInsertCharacter(this, char);
          }
        }
        if (this.p5Instance) {
          this.p5Instance.redraw();
        }
      }
      this.hiddenInput!.value = '';
    });
    
    this.hiddenInput.addEventListener('input', (e) => {
      if (!this.isComposing && this.hiddenInput) {
        const value = this.hiddenInput.value;
        if (value && this.mode === EditorMode.Insert) {
          for (const char of value) {
            this.insertCharacter(char);
          }
          if (this.p5Instance) {
            this.p5Instance.redraw();
          }
        } else if (value && this.mode === EditorMode.TInsert) {
          const handler = this.currentModeHandler as any;
          for (const char of value) {
            if (handler.tInsertCharacter) {
              handler.tInsertCharacter(this, char);
            }
          }
          if (this.p5Instance) {
            this.p5Instance.redraw();
          }
        } else if (value && this.mode === EditorMode.MultiInsert) {
          const handler = this.currentModeHandler as any;
          for (const char of value) {
            if (handler.multiInsertCharacter) {
              handler.multiInsertCharacter(this, char);
            }
          }
          if (this.p5Instance) {
            this.p5Instance.redraw();
          }
        }
        this.hiddenInput.value = '';
      }
    });
    
    this.hiddenInput.focus();
    this.updateInputPosition();
  }

  private updateInputPosition() {
    if (!this.hiddenInput) return;
    
    const line = this.content[this.cursorY] || '';
    const x = this.getTextXPosition(line, this.cursorX);
    const y = this.getRectY(this.cursorY);
    
    this.hiddenInput.style.left = `${x}px`;
    this.hiddenInput.style.top = `${y}px`;
  }

  private waitForP5AndInitialize() {
    // p5 is imported as ES module, so it's available directly
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
        this.updateBuffer();
        
        p.background(0);
        
        this.drawBorder(p);
        this.drawEditorBackground(p);
        this.drawLineNumbers(p);
        this.renderBuffer(p);
        this.drawTMarks(p);
        this.drawFastJumpLabels(p);
        this.drawStatusBar(p);
      };
    };

    // Create p5 instance directly using the imported p5 class
    this.p5Instance = new p5(sketch, this.shadowRoot as unknown as HTMLElement);

    this.cursorBlinkInterval = window.setInterval(() => {
      this.cursorVisible = !this.cursorVisible;
      if (this.p5Instance) {
        this.p5Instance.redraw();
      }
    }, 500);

    window.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  private handleKeyDown(event: KeyboardEvent) {
    const key = event.key;
    
    if (key === 'CapsLock' || key === 'Shift' || key === 'Control' || key === 'Alt' || key === 'Meta') {
      return;
    }
    
    if (this.isComposing || event.isComposing) {
      return;
    }
    
    if (key === 'Process' || key === 'Unidentified') {
      return;
    }
    
    if ((event.metaKey || event.ctrlKey) && key === 'v' && this.mode === EditorMode.Insert) {
      event.preventDefault();
      this.handlePaste();
      return;
    }
    
    this.lastKeyPressed = key;
    
    if (this.currentModeHandler.shouldPreventDefault(key)) {
      event.preventDefault();
    }
    
    this.currentModeHandler.handleKey(key, this);
    
    this.adjustScrollToCursor();
    
    if (this.p5Instance) {
      this.p5Instance.redraw();
    }
  }

  private handleMovement(key: string): boolean {
    if (/^[0-9]$/.test(key)) {
      this.numberPrefix += key;
      return true;
    }
    
    const count = this.numberPrefix ? parseInt(this.numberPrefix, 10) : 1;
    this.numberPrefix = '';
    
    switch (key) {
      case 'j':
      case 'ArrowDown':
        for (let i = 0; i < count; i++) {
          this.moveCursorDown();
        }
        return true;
      case 'k':
      case 'ArrowUp':
        for (let i = 0; i < count; i++) {
          this.moveCursorUp();
        }
        return true;
      case 'h':
      case 'ArrowLeft':
        for (let i = 0; i < count; i++) {
          this.moveCursorLeft();
        }
        return true;
      case 'l':
      case 'ArrowRight':
        for (let i = 0; i < count; i++) {
          this.moveCursorRight();
        }
        return true;
      case '$':
        this.moveCursorToLineEnd();
        return true;
      case '^':
        this.moveCursorToLineStart();
        return true;
      case 'w':
        for (let i = 0; i < count; i++) {
          this.moveToNextWord();
        }
        return true;
      case 'W':
        for (let i = 0; i < count; i++) {
          this.moveToNextWORD();
        }
        return true;
      case 'b':
        for (let i = 0; i < count; i++) {
          this.moveToPreviousWord();
        }
        return true;
      case 'B':
        for (let i = 0; i < count; i++) {
          this.moveToPreviousWORD();
        }
        return true;
      case 'e':
        for (let i = 0; i < count; i++) {
          this.moveToWordEnd();
        }
        return true;
      case 'G':
        this.moveToLastLine();
        return true;
      default:
        return false;
    }
  }




  private moveCursorDown() {
    if (this.cursorY < this.content.length - 1) {
      this.cursorY += 1;
      this.adjustCursorX();
      this.updateInputPosition();
    }
  }

  private moveCursorUp() {
    if (this.cursorY > 0) {
      this.cursorY -= 1;
      this.adjustCursorX();
      this.updateInputPosition();
    }
  }

  private moveCursorLeft() {
    if (this.cursorX > 0) {
      this.cursorX -= 1;
      this.updateInputPosition();
    }
  }

  private moveCursorRight() {
    const currentLine = this.content[this.cursorY] || '';
    const maxPosition = this.mode === EditorMode.Insert ? currentLine.length : currentLine.length - 1;
    if (this.cursorX < maxPosition) {
      this.cursorX += 1;
      this.updateInputPosition();
    }
  }

  private moveCursorToLineEnd() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0) {
      this.cursorX = currentLine.length - 1;
      this.updateInputPosition();
    }
  }

  private moveCursorToLineStart() {
    const currentLine = this.content[this.cursorY] || '';
    let firstNonSpace = 0;
    
    for (let i = 0; i < currentLine.length; i++) {
      if (currentLine[i] !== ' ' && currentLine[i] !== '\t') {
        firstNonSpace = i;
        break;
      }
    }
    
    this.cursorX = firstNonSpace;
    this.updateInputPosition();
  }

  private moveToFirstLine() {
    this.cursorY = 0;
    this.moveCursorToLineStart();
  }

  private moveToLastLine() {
    this.cursorY = Math.max(0, this.content.length - 1);
    this.adjustCursorX();
    this.updateInputPosition();
  }

  private jumpToMatchingBracket() {
    const currentLine = this.content[this.cursorY] || '';
    const currentChar = currentLine[this.cursorX];
    
    if (!currentChar) {
      return;
    }
    
    const openBrackets: { [key: string]: string } = {
      '[': ']',
      '{': '}',
      '(': ')',
      '"': '"',
      "'": "'",
      '`': '`',
      '<': '>',
    };
    
    const closeBrackets: { [key: string]: string } = {
      ']': '[',
      '}': '{',
      ')': '(',
      '"': '"',
      "'": "'",
      '`': '`',
      '>': '<',
    };
    
    let matchPos: { y: number; x: number } | null = null;
    
    if (openBrackets[currentChar]) {
      matchPos = this.findMatchingBracketForward(currentChar, openBrackets[currentChar]);
    } else if (closeBrackets[currentChar]) {
      matchPos = this.findMatchingBracketBackward(closeBrackets[currentChar], currentChar);
    }
    
    if (matchPos) {
      this.cursorY = matchPos.y;
      this.cursorX = matchPos.x;
      this.updateInputPosition();
    }
  }

  private findMatchingBracketForward(openChar: string, closeChar: string): { y: number; x: number } | null {
    let depth = 0;
    const isQuote = openChar === closeChar;
    
    for (let y = this.cursorY; y < this.content.length; y++) {
      const line = this.content[y];
      const startX = y === this.cursorY ? this.cursorX : 0;
      
      for (let x = startX; x < line.length; x++) {
        if (this.isEscaped(line, x)) {
          continue;
        }
        
        const char = line[x];
        
        if (isQuote) {
          if (char === openChar) {
            if (y === this.cursorY && x === this.cursorX) {
              depth++;
            } else if (depth === 1) {
              return { y, x };
            } else {
              depth++;
            }
          }
        } else {
          if (char === openChar) {
            depth++;
          } else if (char === closeChar) {
            depth--;
            if (depth === 0) {
              return { y, x };
            }
          }
        }
      }
    }
    
    return null;
  }

  private findMatchingBracketBackward(openChar: string, closeChar: string): { y: number; x: number } | null {
    let depth = 0;
    const isQuote = openChar === closeChar;
    
    for (let y = this.cursorY; y >= 0; y--) {
      const line = this.content[y];
      const startX = y === this.cursorY ? this.cursorX : line.length - 1;
      
      for (let x = startX; x >= 0; x--) {
        if (this.isEscaped(line, x)) {
          continue;
        }
        
        const char = line[x];
        
        if (isQuote) {
          if (char === closeChar) {
            if (y === this.cursorY && x === this.cursorX) {
              depth++;
            } else if (depth === 1) {
              return { y, x };
            } else {
              depth++;
            }
          }
        } else {
          if (char === closeChar) {
            depth++;
          } else if (char === openChar) {
            depth--;
            if (depth === 0) {
              return { y, x };
            }
          }
        }
      }
    }
    
    return null;
  }

  private isEscaped(line: string, position: number): boolean {
    if (position === 0) {
      return false;
    }
    
    let backslashCount = 0;
    for (let i = position - 1; i >= 0 && line[i] === '\\'; i--) {
      backslashCount++;
    }
    
    return backslashCount % 2 === 1;
  }

  private adjustScrollToCursor() {
    if (this.cursorY < this.scrollOffsetY) {
      this.scrollOffsetY = this.cursorY;
    }
    
    if (this.cursorY >= this.scrollOffsetY + this.bufferHeight) {
      this.scrollOffsetY = this.cursorY - this.bufferHeight + 1;
    }
    
    if (this.cursorX < this.scrollOffsetX) {
      this.scrollOffsetX = this.cursorX;
    }
    
    const currentLine = this.content[this.cursorY] || '';
    const availablePixelWidth = (800 - 60) - (this.baseCharWidth * 2);
    
    let linePixelWidth = 0;
    for (let i = 0; i < currentLine.length; i++) {
      linePixelWidth += this.isFullWidthChar(currentLine[i]) ? this.baseCharWidth * 2 : this.baseCharWidth;
    }
    
    if (linePixelWidth <= availablePixelWidth) {
      this.scrollOffsetX = 0;
      return;
    }
    
    let scrollEndX = this.scrollOffsetX;
    let accumulatedWidth = 0;
    
    while (scrollEndX < currentLine.length && accumulatedWidth < availablePixelWidth) {
      accumulatedWidth += this.isFullWidthChar(currentLine[scrollEndX]) ? this.baseCharWidth * 2 : this.baseCharWidth;
      scrollEndX++;
    }
    
    if (this.cursorX >= scrollEndX) {
      console.log(`[Scroll] Horizontal scroll triggered: cursorX=${this.cursorX}, scrollEndX=${scrollEndX}, scrollOffsetX=${this.scrollOffsetX}`);
      
      while (this.cursorX >= scrollEndX && this.scrollOffsetX < this.cursorX) {
        this.scrollOffsetX++;
        
        scrollEndX = this.scrollOffsetX;
        accumulatedWidth = 0;
        
        while (scrollEndX < currentLine.length && accumulatedWidth < availablePixelWidth) {
          accumulatedWidth += this.isFullWidthChar(currentLine[scrollEndX]) ? this.baseCharWidth * 2 : this.baseCharWidth;
          scrollEndX++;
        }
      }
      
      console.log(`[Scroll] New scrollOffsetX=${this.scrollOffsetX}, new scrollEndX=${scrollEndX}`);
    }
  }

  private isWordChar(char: string): boolean {
    return /\w/.test(char);
  }

  private isChinese(char: string): boolean {
    return /[\u4e00-\u9fa5]/.test(char);
  }

  private getCharType(char: string): 'word' | 'chinese' | 'space' {
    if (this.isChinese(char)) {
      return 'chinese';
    } else if (this.isWordChar(char)) {
      return 'word';
    } else {
      return 'space';
    }
  }

  private getInnerWordRange(): { startX: number; endX: number; y: number } | null {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length === 0) {
      return null;
    }
    
    const cursorChar = currentLine[this.cursorX];
    if (!cursorChar) {
      return null;
    }
    
    const charType = this.getCharType(cursorChar);
    if (charType === 'space') {
      return null;
    }
    
    let startX = this.cursorX;
    let endX = this.cursorX;
    
    while (startX > 0 && this.getCharType(currentLine[startX - 1]) === charType) {
      startX--;
    }
    
    while (endX < currentLine.length - 1 && this.getCharType(currentLine[endX + 1]) === charType) {
      endX++;
    }
    
    return { startX, endX, y: this.cursorY };
  }


  private deleteInnerWord() {
    const range = this.getInnerWordRange();
    if (!range) {
      return;
    }
    
    const currentLine = this.content[range.y];
    const beforeWord = currentLine.substring(0, range.startX);
    const afterWord = currentLine.substring(range.endX + 1);
    
    this.content[range.y] = beforeWord + afterWord;
    
    this.cursorX = range.startX;
    if (this.cursorX >= this.content[range.y].length && this.content[range.y].length > 0) {
      this.cursorX = this.content[range.y].length - 1;
    }
    if (this.cursorX < 0) {
      this.cursorX = 0;
    }
  }

  private deleteInnerQuote(quoteChar: string) {
    const range = this.getInnerQuoteRange(quoteChar);
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = this.content[range.startY];
      const beforeQuote = currentLine.substring(0, range.startX + 1);
      const afterQuote = currentLine.substring(range.endX);
      
      this.content[range.startY] = beforeQuote + afterQuote;
      
      this.cursorY = range.startY;
      this.cursorX = range.startX + 1;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    } else {
      const firstLine = this.content[range.startY];
      const lastLine = this.content[range.endY];
      
      const beforeQuote = firstLine.substring(0, range.startX + 1);
      const afterQuote = lastLine.substring(range.endX);
      
      this.content[range.startY] = beforeQuote + afterQuote;
      this.content.splice(range.startY + 1, range.endY - range.startY);
      
      this.cursorY = range.startY;
      this.cursorX = range.startX + 1;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    }
  }

  private deleteInnerBracket() {
    const range = this.getInnerBracketRange();
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = this.content[range.startY];
      const beforeBracket = currentLine.substring(0, range.startX + 1);
      const afterBracket = currentLine.substring(range.endX);
      
      this.content[range.startY] = beforeBracket + afterBracket;
      
      this.cursorY = range.startY;
      this.cursorX = range.startX + 1;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    } else {
      const firstLine = this.content[range.startY];
      const lastLine = this.content[range.endY];
      
      const beforeBracket = firstLine.substring(0, range.startX + 1);
      const afterBracket = lastLine.substring(range.endX);
      
      this.content[range.startY] = beforeBracket + afterBracket;
      this.content.splice(range.startY + 1, range.endY - range.startY);
      
      this.cursorY = range.startY;
      this.cursorX = range.startX + 1;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    }
  }

  private deleteAroundBracket(openChar: string, closeChar: string) {
    const range = this.findInnerBracketRange(openChar, closeChar);
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = this.content[range.startY];
      const beforeBracket = currentLine.substring(0, range.startX);
      const afterBracket = currentLine.substring(range.endX + 1);
      
      this.content[range.startY] = beforeBracket + afterBracket;
      
      this.cursorY = range.startY;
      this.cursorX = range.startX;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    } else {
      const firstLine = this.content[range.startY];
      const lastLine = this.content[range.endY];
      
      const beforeBracket = firstLine.substring(0, range.startX);
      const afterBracket = lastLine.substring(range.endX + 1);
      
      this.content[range.startY] = beforeBracket + afterBracket;
      this.content.splice(range.startY + 1, range.endY - range.startY);
      
      this.cursorY = range.startY;
      this.cursorX = range.startX;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    }
  }

  private deleteAroundQuote(quoteChar: string) {
    const range = this.getInnerQuoteRange(quoteChar);
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = this.content[range.startY];
      const beforeQuote = currentLine.substring(0, range.startX);
      const afterQuote = currentLine.substring(range.endX + 1);
      
      this.content[range.startY] = beforeQuote + afterQuote;
      
      this.cursorY = range.startY;
      this.cursorX = range.startX;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    } else {
      const firstLine = this.content[range.startY];
      const lastLine = this.content[range.endY];
      
      const beforeQuote = firstLine.substring(0, range.startX);
      const afterQuote = lastLine.substring(range.endX + 1);
      
      this.content[range.startY] = beforeQuote + afterQuote;
      this.content.splice(range.startY + 1, range.endY - range.startY);
      
      this.cursorY = range.startY;
      this.cursorX = range.startX;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    }
  }

  private deleteAroundAnyBracket() {
    const range = this.getInnerBracketRange();
    if (!range) {
      return;
    }
    
    if (range.startY === range.endY) {
      const currentLine = this.content[range.startY];
      const beforeBracket = currentLine.substring(0, range.startX);
      const afterBracket = currentLine.substring(range.endX + 1);
      
      this.content[range.startY] = beforeBracket + afterBracket;
      
      this.cursorY = range.startY;
      this.cursorX = range.startX;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    } else {
      const firstLine = this.content[range.startY];
      const lastLine = this.content[range.endY];
      
      const beforeBracket = firstLine.substring(0, range.startX);
      const afterBracket = lastLine.substring(range.endX + 1);
      
      this.content[range.startY] = beforeBracket + afterBracket;
      this.content.splice(range.startY + 1, range.endY - range.startY);
      
      this.cursorY = range.startY;
      this.cursorX = range.startX;
      if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
        this.cursorX = this.content[this.cursorY].length - 1;
      }
    }
  }


  private getInnerQuoteRange(quoteChar: string): { startY: number; startX: number; endY: number; endX: number } | null {
    const range = this.findMultiLineQuoteRange(quoteChar);
    if (!range) {
      return null;
    }
    
    if (range.startY === range.endY && range.startX === range.endX) {
      return null;
    }
    
    return range;
  }

  private getInnerBracketRange(): { startY: number; startX: number; endY: number; endX: number } | null {
    const bracketPairs: Array<[string, string]> = [
      ['[', ']'],
      ['{', '}'],
      ['(', ')'],
      ['"', '"'],
      ["'", "'"],
      ['`', '`'],
      ['<', '>'],
    ];
    
    let closestRange: { startY: number; startX: number; endY: number; endX: number; distance: number } | null = null;
    
    for (const [openChar, closeChar] of bracketPairs) {
      const range = this.findInnerBracketRange(openChar, closeChar);
      if (range) {
        const distance = this.calculateDistanceFromCursor(range.startY, range.startX, range.endY, range.endX);
        if (!closestRange || distance < closestRange.distance) {
          closestRange = { ...range, distance };
        }
      }
    }
    
    if (!closestRange) {
      return null;
    }
    
    return {
      startY: closestRange.startY,
      startX: closestRange.startX,
      endY: closestRange.endY,
      endX: closestRange.endX,
    };
  }

  private findInnerBracketRange(openChar: string, closeChar: string): { startY: number; startX: number; endY: number; endX: number } | null {
    const isQuote = openChar === closeChar;
    
    if (isQuote) {
      return this.findMultiLineQuoteRange(openChar);
    }
    
    let openPos = this.findOpeningBracketBeforeCursor(openChar, closeChar);
    if (!openPos) {
      return null;
    }
    
    let closePos = this.findClosingBracketFromPosition(openChar, closeChar, openPos.y, openPos.x);
    if (!closePos) {
      return null;
    }
    
    if (this.isCursorInRange(openPos.y, openPos.x, closePos.y, closePos.x)) {
      return {
        startY: openPos.y,
        startX: openPos.x,
        endY: closePos.y,
        endX: closePos.x,
      };
    }
    
    return null;
  }

  private findOpeningBracketBeforeCursor(openChar: string, closeChar: string): { y: number; x: number } | null {
    let depth = 0;
    let skipFirst = false;
    
    const currentLine = this.content[this.cursorY] || '';
    if (this.cursorX < currentLine.length && currentLine[this.cursorX] === closeChar && !this.isEscaped(currentLine, this.cursorX)) {
      skipFirst = true;
    }
    
    for (let y = this.cursorY; y >= 0; y--) {
      const line = this.content[y];
      const endX = y === this.cursorY ? this.cursorX : line.length - 1;
      
      for (let x = endX; x >= 0; x--) {
        if (this.isEscaped(line, x)) {
          continue;
        }
        
        const char = line[x];
        
        if (char === closeChar) {
          if (skipFirst && y === this.cursorY && x === this.cursorX) {
            skipFirst = false;
            continue;
          }
          depth++;
        } else if (char === openChar) {
          if (depth === 0) {
            return { y, x };
          }
          depth--;
        }
      }
    }
    
    return null;
  }

  private findClosingBracketFromPosition(openChar: string, closeChar: string, startY: number, startX: number): { y: number; x: number } | null {
    let depth = 0;
    
    for (let y = startY; y < this.content.length; y++) {
      const line = this.content[y];
      const beginX = y === startY ? startX : 0;
      
      for (let x = beginX; x < line.length; x++) {
        if (this.isEscaped(line, x)) {
          continue;
        }
        
        const char = line[x];
        
        if (char === openChar) {
          depth++;
        } else if (char === closeChar) {
          depth--;
          if (depth === 0) {
            return { y, x };
          }
        }
      }
    }
    
    return null;
  }

  private calculateDistanceFromCursor(startY: number, startX: number, endY: number, endX: number): number {
    const distToStart = Math.abs(this.cursorY - startY) * 1000 + Math.abs(this.cursorX - startX);
    const distToEnd = Math.abs(this.cursorY - endY) * 1000 + Math.abs(this.cursorX - endX);
    return Math.min(distToStart, distToEnd);
  }

  private findMultiLineQuoteRange(quoteChar: string): { startY: number; startX: number; endY: number; endX: number } | null {
    let startY = -1;
    let startX = -1;
    let endY = -1;
    let endX = -1;
    let inQuote = false;
    
    for (let y = 0; y < this.content.length; y++) {
      const line = this.content[y];
      
      for (let x = 0; x < line.length; x++) {
        if (line[x] === quoteChar) {
          let isEscaped = false;
          if (x > 0 && line[x - 1] === '\\') {
            let backslashCount = 0;
            for (let j = x - 1; j >= 0 && line[j] === '\\'; j--) {
              backslashCount++;
            }
            isEscaped = backslashCount % 2 === 1;
          }
          
          if (!isEscaped) {
            if (!inQuote) {
              startY = y;
              startX = x;
              inQuote = true;
            } else {
              endY = y;
              endX = x;
              
              if (this.isCursorInRange(startY, startX, endY, endX)) {
                return { startY, startX, endY, endX };
              }
              
              inQuote = false;
            }
          }
        }
      }
    }
    
    return null;
  }

  private isCursorInRange(startY: number, startX: number, endY: number, endX: number): boolean {
    if (this.cursorY < startY || this.cursorY > endY) {
      return false;
    }
    
    if (this.cursorY === startY && this.cursorX < startX) {
      return false;
    }
    
    if (this.cursorY === endY && this.cursorX > endX) {
      return false;
    }
    
    return true;
  }

  private deleteWord() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length === 0 || this.cursorX >= currentLine.length) {
      return;
    }
    
    const startX = this.cursorX;
    let endX = this.cursorX;
    
    const currentChar = currentLine[this.cursorX];
    const charType = this.getCharType(currentChar);
    
    if (charType === 'space') {
      while (endX < currentLine.length && this.getCharType(currentLine[endX]) === 'space') {
        endX++;
      }
    } else {
      while (endX < currentLine.length && this.getCharType(currentLine[endX]) === charType) {
        endX++;
      }
      
      while (endX < currentLine.length && this.getCharType(currentLine[endX]) === 'space') {
        endX++;
      }
    }
    
    const beforeWord = currentLine.substring(0, startX);
    const afterWord = currentLine.substring(endX);
    
    this.content[this.cursorY] = beforeWord + afterWord;

    this.cursorX = startX;
    if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
      this.cursorX = this.content[this.cursorY].length - 1;
    }
    if (this.cursorX < 0) {
      this.cursorX = 0;
    }
  }

  private deleteToWordEnd() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length === 0 || this.cursorX >= currentLine.length) {
      return;
    }
    
    const currentChar = currentLine[this.cursorX];
    const charType = this.getCharType(currentChar);
    
    if (charType === 'space') {
      return;
    }
    
    const startX = this.cursorX;
    let endX = this.cursorX;
    
    while (endX < currentLine.length && this.getCharType(currentLine[endX]) === charType) {
      endX++;
    }
    
    const beforeWord = currentLine.substring(0, startX);
    const afterWord = currentLine.substring(endX);
    
    this.content[this.cursorY] = beforeWord + afterWord;
    
    this.cursorX = startX;
    if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
      this.cursorX = this.content[this.cursorY].length - 1;
    }
    if (this.cursorX < 0) {
      this.cursorX = 0;
    }
  }

  private deleteLinesDown(count: number) {
    const startY = this.cursorY;
    const endY = Math.min(startY + count, this.content.length - 1);
    const linesToDelete = endY - startY + 1;
    
    this.content.splice(startY, linesToDelete);
    
    if (this.content.length === 0) {
      this.content = [''];
    }
    
    if (this.cursorY >= this.content.length) {
      this.cursorY = this.content.length - 1;
    }
    
    this.cursorX = 0;
    this.adjustCursorX();
  }

  private deleteLinesUp(count: number) {
    const endY = this.cursorY;
    const startY = Math.max(endY - count, 0);
    const linesToDelete = endY - startY + 1;
    
    this.content.splice(startY, linesToDelete);
    
    if (this.content.length === 0) {
      this.content = [''];
    }
    
    this.cursorY = startY;
    if (this.cursorY >= this.content.length) {
      this.cursorY = this.content.length - 1;
    }
    
    this.cursorX = 0;
    this.adjustCursorX();
  }

  private moveToNextWord() {
    const currentLine = this.content[this.cursorY] || '';
    
    if (this.cursorX >= currentLine.length - 1) {
      if (this.cursorY < this.content.length - 1) {
        this.cursorY += 1;
        this.cursorX = 0;
        const nextLine = this.content[this.cursorY] || '';
        for (let i = 0; i < nextLine.length; i++) {
          if (this.getCharType(nextLine[i]) !== 'space') {
            this.cursorX = i;
            break;
          }
        }
      }
      this.updateInputPosition();
      return;
    }
    
    const currentType = this.getCharType(currentLine[this.cursorX]);
    let skipCurrent = true;
    
    for (let i = this.cursorX + 1; i < currentLine.length; i++) {
      const charType = this.getCharType(currentLine[i]);
      
      if (charType === 'space') {
        skipCurrent = false;
        continue;
      }
      
      if (skipCurrent && charType === currentType) {
        continue;
      }
      
      this.cursorX = i;
      this.updateInputPosition();
      return;
    }
    
    this.cursorX = currentLine.length - 1;
    this.updateInputPosition();
  }

  private moveToWordEnd() {
    const currentLine = this.content[this.cursorY] || '';
    
    if (this.cursorX >= currentLine.length) {
      return;
    }
    
    const currentChar = currentLine[this.cursorX];
    const currentType = this.getCharType(currentChar);
    
    if (currentType === 'space') {
      return;
    }
    
    let startPos = this.cursorX;
    const nextChar = currentLine[this.cursorX + 1];
    
    if (nextChar && this.getCharType(nextChar) !== currentType) {
      startPos = this.cursorX + 1;
      
      while (startPos < currentLine.length && this.getCharType(currentLine[startPos]) === 'space') {
        startPos++;
      }
      
      if (startPos >= currentLine.length) {
        return;
      }
    }
    
    const newType = this.getCharType(currentLine[startPos]);
    
    for (let i = startPos + 1; i < currentLine.length; i++) {
      const charType = this.getCharType(currentLine[i]);
      
      if (charType !== newType) {
        this.cursorX = i - 1;
        this.updateInputPosition();
        return;
      }
    }
    
    this.cursorX = currentLine.length - 1;
    this.updateInputPosition();
  }

  private moveToPreviousWord() {
    if (this.cursorX === 0) {
      if (this.cursorY > 0) {
        this.cursorY -= 1;
        const prevLine = this.content[this.cursorY] || '';
        this.cursorX = Math.max(0, prevLine.length - 1);
        
        for (let i = this.cursorX; i >= 0; i--) {
          const charType = this.getCharType(prevLine[i]);
          if (charType !== 'space') {
            for (let j = i; j >= 0; j--) {
              if (this.getCharType(prevLine[j]) !== charType) {
                this.cursorX = j + 1;
                this.updateInputPosition();
                return;
              }
            }
            this.cursorX = 0;
            this.updateInputPosition();
            return;
          }
        }
      }
      this.updateInputPosition();
      return;
    }
    
    const currentLine = this.content[this.cursorY] || '';
    const currentType = this.getCharType(currentLine[this.cursorX]);
    let skipCurrent = currentType !== 'space';
    
    for (let i = this.cursorX - 1; i >= 0; i--) {
      const charType = this.getCharType(currentLine[i]);
      
      if (charType === 'space') {
        skipCurrent = false;
        continue;
      }
      
      if (skipCurrent && charType === currentType) {
        continue;
      }
      
      for (let j = i; j >= 0; j--) {
        if (this.getCharType(currentLine[j]) !== charType) {
          this.cursorX = j + 1;
          this.updateInputPosition();
          return;
        }
      }
      this.cursorX = 0;
      this.updateInputPosition();
      return;
    }
    
    this.cursorX = 0;
    this.updateInputPosition();
  }

  private moveToNextWORD() {
    const currentLine = this.content[this.cursorY] || '';
    
    if (this.cursorX >= currentLine.length - 1) {
      if (this.cursorY < this.content.length - 1) {
        this.cursorY += 1;
        this.cursorX = 0;
        const nextLine = this.content[this.cursorY] || '';
        for (let i = 0; i < nextLine.length; i++) {
          if (nextLine[i] !== ' ' && nextLine[i] !== '\t') {
            this.cursorX = i;
            break;
          }
        }
      }
      this.updateInputPosition();
      return;
    }
    
    const isCurrentSpace = currentLine[this.cursorX] === ' ' || currentLine[this.cursorX] === '\t';
    let foundNonSpace = false;
    
    for (let i = this.cursorX + 1; i < currentLine.length; i++) {
      const isSpace = currentLine[i] === ' ' || currentLine[i] === '\t';
      
      if (isSpace) {
        foundNonSpace = true;
      } else if (foundNonSpace || isCurrentSpace) {
        this.cursorX = i;
        this.updateInputPosition();
        return;
      }
    }
    
    this.cursorX = currentLine.length - 1;
    this.updateInputPosition();
  }

  private moveToPreviousWORD() {
    if (this.cursorX === 0) {
      if (this.cursorY > 0) {
        this.cursorY -= 1;
        const prevLine = this.content[this.cursorY] || '';
        this.cursorX = Math.max(0, prevLine.length - 1);
        
        for (let i = this.cursorX; i >= 0; i--) {
          const isSpace = prevLine[i] === ' ' || prevLine[i] === '\t';
          if (!isSpace) {
            for (let j = i; j >= 0; j--) {
              const isSpaceJ = prevLine[j] === ' ' || prevLine[j] === '\t';
              if (isSpaceJ) {
                this.cursorX = j + 1;
                this.updateInputPosition();
                return;
              }
            }
            this.cursorX = 0;
            this.updateInputPosition();
            return;
          }
        }
      }
      this.updateInputPosition();
      return;
    }
    
    const currentLine = this.content[this.cursorY] || '';
    const isCurrentSpace = currentLine[this.cursorX] === ' ' || currentLine[this.cursorX] === '\t';
    let skipCurrent = !isCurrentSpace;
    
    for (let i = this.cursorX - 1; i >= 0; i--) {
      const isSpace = currentLine[i] === ' ' || currentLine[i] === '\t';
      
      if (isSpace) {
        skipCurrent = false;
        continue;
      }
      
      if (skipCurrent) {
        continue;
      }
      
      for (let j = i; j >= 0; j--) {
        const isSpaceJ = currentLine[j] === ' ' || currentLine[j] === '\t';
        if (isSpaceJ) {
          this.cursorX = j + 1;
          this.updateInputPosition();
          return;
        }
      }
      this.cursorX = 0;
      this.updateInputPosition();
      return;
    }
    
    this.cursorX = 0;
    this.updateInputPosition();
  }

  private adjustCursorX() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0 && this.cursorX >= currentLine.length) {
      this.cursorX = currentLine.length - 1;
    }
  }

  private adjustCursorForNormalMode() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0 && this.cursorX >= currentLine.length) {
      this.cursorX = currentLine.length - 1;
    }
  }

  private insertLineBelow() {
    this.content.splice(this.cursorY + 1, 0, '');
    this.cursorY += 1;
    this.cursorX = 0;
    this.mode = EditorMode.Insert;
    this.updateInputPosition();
  }


  private deleteMultiLineSelection(startY: number, endY: number, startX: number, endX: number) {
    const firstPart = this.content[startY].slice(0, startX);
    const lastPart = this.content[endY].slice(endX + 1);
    this.content[startY] = firstPart + lastPart;
    this.content.splice(startY + 1, endY - startY);
    this.cursorX = startX;
    this.cursorY = startY;
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
    this.updateInputPosition();
  }

  private handleEnter() {
    const currentLine = this.content[this.cursorY];
    const lineBeforeCursor = currentLine.substring(0, this.cursorX);
    const lineAfterCursor = currentLine.substring(this.cursorX);
    
    this.content[this.cursorY] = lineBeforeCursor;
    this.content.splice(this.cursorY + 1, 0, lineAfterCursor);
    
    this.cursorY += 1;
    this.cursorX = 0;
    this.updateInputPosition();
  }

  private insertCharacter(char: string) {
    const currentLine = this.content[this.cursorY];
    this.content[this.cursorY] = 
      currentLine.substring(0, this.cursorX) + 
      char + 
      currentLine.substring(this.cursorX);
    this.cursorX += 1;
    this.updateInputPosition();
  }

  private async handlePaste() {
    try {
      const text = await navigator.clipboard.readText();
      this.insertText(text);
      if (this.p5Instance) {
        this.p5Instance.redraw();
      }
    } catch (err) {
      console.error('Failed to read clipboard:', err);
    }
  }

  private async pasteAfterCursor() {
    try {
      const text = await navigator.clipboard.readText();
      this.pasteTextAfterCursor(text);
      if (this.p5Instance) {
        this.p5Instance.redraw();
      }
    } catch (err) {
      console.error('Failed to read clipboard:', err);
    }
  }

  private insertText(text: string) {
    const lines = text.split('\n');
    
    if (lines.length === 1) {
      for (const char of text) {
        this.insertCharacter(char);
      }
    } else {
      const currentLine = this.content[this.cursorY];
      const beforeCursor = currentLine.substring(0, this.cursorX);
      const afterCursor = currentLine.substring(this.cursorX);
      
      this.content[this.cursorY] = beforeCursor + lines[0];
      
      for (let i = 1; i < lines.length - 1; i++) {
        this.cursorY += 1;
        this.content.splice(this.cursorY, 0, lines[i]);
      }
      
      if (lines.length > 1) {
        this.cursorY += 1;
        this.content.splice(this.cursorY, 0, lines[lines.length - 1] + afterCursor);
        this.cursorX = lines[lines.length - 1].length;
      }
      
      this.updateInputPosition();
    }
  }

  private pasteTextAfterCursor(text: string) {
    const currentLine = this.content[this.cursorY] || '';
    const lines = text.split('\n');
    
    if (lines.length === 1) {
      let insertPosition: number;
      if (currentLine.length === 0) {
        insertPosition = 0;
      } else {
        insertPosition = Math.min(this.cursorX + 1, currentLine.length);
      }
      
      const beforeInsert = currentLine.substring(0, insertPosition);
      const afterInsert = currentLine.substring(insertPosition);
      
      this.content[this.cursorY] = beforeInsert + text + afterInsert;
      
      if (currentLine.length === 0) {
        this.cursorX = text.length - 1;
      } else {
        this.cursorX = insertPosition + text.length - 1;
      }
    } else {
      let insertPosition: number;
      if (currentLine.length === 0) {
        insertPosition = 0;
      } else {
        insertPosition = Math.min(this.cursorX + 1, currentLine.length);
      }
      
      const beforeInsert = currentLine.substring(0, insertPosition);
      const afterInsert = currentLine.substring(insertPosition);
      
      this.content[this.cursorY] = beforeInsert + lines[0];
      
      for (let i = 1; i < lines.length - 1; i++) {
        this.cursorY += 1;
        this.content.splice(this.cursorY, 0, lines[i]);
      }
      
      if (lines.length > 1) {
        this.cursorY += 1;
        const lastLine = lines[lines.length - 1];
        this.content.splice(this.cursorY, 0, lastLine + afterInsert);
        this.cursorX = lastLine.length > 0 ? lastLine.length - 1 : 0;
      }
    }
    
    this.updateInputPosition();
  }

  private saveHistory(cursorPos: { cursorX: number; cursorY: number } | null = null) {
    const saveCursorX = cursorPos !== null ? cursorPos.cursorX : this.cursorX;
    const saveCursorY = cursorPos !== null ? cursorPos.cursorY : this.cursorY;
    
    this.history.push({
      content: JSON.parse(JSON.stringify(this.content)),
      cursorX: saveCursorX,
      cursorY: saveCursorY
    });
    
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
    } else {
      this.historyIndex++;
    }
  }

  private undo() {
    if (this.historyIndex >= 0) {
      const state = this.history[this.historyIndex];
      this.historyIndex--;
      this.content = JSON.parse(JSON.stringify(state.content));
      this.cursorX = state.cursorX;
      this.cursorY = state.cursorY;
      this.updateInputPosition();
      if (this.p5Instance) {
        this.p5Instance.redraw();
      }
    }
  }

  resetHistory() {
    this.history = [];
    this.historyIndex = -1;
  }


  disconnectedCallback() {
    if (this.cursorBlinkInterval) {
      clearInterval(this.cursorBlinkInterval);
    }
    if (this.p5Instance) {
      this.p5Instance.remove();
    }
    if (this.hiddenInput) {
      this.hiddenInput.remove();
    }
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
    
    p.textAlign(p.RIGHT, p.TOP);
    
    for (let bufferY = 0; bufferY < this.bufferHeight; bufferY++) {
      const contentY = bufferY + this.scrollOffsetY;
      
      if (contentY >= this.content.length) {
        break;
      }
      
      const isCursorLine = contentY === this.cursorY;
      
      if (isCursorLine) {
        p.fill(255, 255, 0);
        const lineNum = (contentY + 1).toString();
        p.text(lineNum, 45, this.getTextY(bufferY));
      } else {
        p.fill(100, 100, 100);
        const relativeNum = Math.abs(contentY - this.cursorY).toString();
        p.text(relativeNum, 45, this.getTextY(bufferY));
      }
    }
    
    p.textAlign(p.LEFT, p.TOP);
  }

  private renderBuffer(p: p5) {
    for (let bufferY = 0; bufferY < this.bufferHeight && bufferY < this.buffer.length; bufferY++) {
      const contentY = bufferY + this.scrollOffsetY;
      const line = this.content[contentY] || '';
      
      for (let bufferX = 0; bufferX < this.bufferWidth && bufferX < this.buffer[bufferY].length; bufferX++) {
        const contentX = bufferX + this.scrollOffsetX;
        const cell = this.buffer[bufferY][bufferX];
        const char = line[contentX] || ' ';
        const charWidth = this.getCharWidth(char);
        const screenX = this.getTextXPosition(line.substring(this.scrollOffsetX), bufferX);
        
        if (cell.background[0] !== 0 || cell.background[1] !== 0 || cell.background[2] !== 0) {
          p.fill(cell.background[0], cell.background[1], cell.background[2]);
          p.rect(screenX, this.getRectY(bufferY), charWidth, this.lineHeight);
        }
        
        p.fill(cell.foreground[0], cell.foreground[1], cell.foreground[2]);
        p.text(cell.char, screenX, this.getTextY(bufferY));
      }
    }
    
    this.drawInsertCursor(p);
  }

  private drawInsertCursor(p: p5) {
    if ((this.mode !== EditorMode.Insert && this.mode !== EditorMode.MultiInsert && this.mode !== EditorMode.TInsert) || !this.cursorVisible) {
      return;
    }
    
    const bufferY = this.cursorY - this.scrollOffsetY;
    const bufferX = this.cursorX - this.scrollOffsetX;
    
    if (bufferY < 0 || bufferY >= this.bufferHeight || bufferX < 0 || bufferX >= this.bufferWidth) {
      return;
    }
    
    const line = this.content[this.cursorY] || '';
    const visibleLine = line.substring(this.scrollOffsetX);
    const screenX = this.getTextXPosition(visibleLine, bufferX);
    const screenY = this.getRectY(bufferY);
    
    p.stroke(255);
    p.strokeWeight(2);
    p.line(screenX, screenY, screenX, screenY + this.lineHeight);
    p.noStroke();
  }

  private drawFastJumpLabels(p: p5) {
    if (this.mode !== EditorMode.FastMatch || this.fastJumpMatches.length === 0) {
      return;
    }
    
    for (const match of this.fastJumpMatches) {
      const bufferY = match.y - this.scrollOffsetY;
      const bufferX = match.x - this.scrollOffsetX;
      
      if (bufferY < 0 || bufferY >= this.bufferHeight || bufferX < 0 || bufferX >= this.bufferWidth) {
        continue;
      }
      
      const line = this.content[match.y] || '';
      const visibleLine = line.substring(this.scrollOffsetX);
      const screenX = this.getTextXPosition(visibleLine, bufferX);
      const screenY = this.getRectY(bufferY);
      
      const labelWidth = match.label.length * this.baseCharWidth + 4;
      const labelHeight = this.lineHeight;
      
      p.fill(255, 200, 0);
      p.rect(screenX, screenY, labelWidth, labelHeight);
      
      p.fill(0, 0, 0);
      p.text(match.label, screenX + 2, this.getTextY(bufferY));
    }
  }

  private drawTMarks(p: p5) {
    if (this.tMarks.length === 0) {
      return;
    }
    
    for (const mark of this.tMarks) {
      const bufferY = mark.y - this.scrollOffsetY;
      const bufferX = mark.x - this.scrollOffsetX;
      
      if (bufferY < 0 || bufferY >= this.bufferHeight || bufferX < 0 || bufferX >= this.bufferWidth) {
        continue;
      }
      
      const line = this.content[mark.y] || '';
      const visibleLine = line.substring(this.scrollOffsetX);
      const screenX = this.getTextXPosition(visibleLine, bufferX);
      const screenY = this.getRectY(bufferY);
      
      const markWidth = this.baseCharWidth * 0.5;
      const markHeight = this.lineHeight;
      const markX = screenX - markWidth - 2;
      
      p.fill(255, 100, 100);
      p.rect(markX, screenY, markWidth, markHeight);
      
      p.fill(255, 255, 255);
      p.textSize(12);
      p.text('t', markX, screenY + 2);
      p.textSize(16);
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
    let statusText = `Mode: ${this.mode} | Line: ${this.cursorY + 1}, Col: ${this.getDisplayColumn() + 1}`;
    if (this.lastKeyPressed) {
      statusText += ` | Key: "${this.lastKeyPressed}"`;
    }
    
    p.text(
      statusText,
      10,
      statusY + 3 // 計算垂直居中位置
    );
  }


  private enterMultiInsertMode(moveNext: boolean = false) {
    if (moveNext && this.currentMatchIndex >= 0 && this.searchMatches.length > 0) {
      const match = this.searchMatches[this.currentMatchIndex];
      const matchEndX = match.x + this.searchKeyword.length;
      if (this.cursorX < matchEndX) {
        this.cursorX++;
      }
    }
    
    this.saveHistory();
    this.mode = EditorMode.MultiInsert;
    this.hiddenInput?.focus();
  }


  private enterInsertMode() {
    if (this.tMarks.length > 0) {
      this.mode = EditorMode.TInsert;
      const lastMark = this.tMarks[this.tMarks.length - 1];
      this.cursorY = lastMark.y;
      this.cursorX = lastMark.x;
      this.updateInputPosition();
    } else {
      this.mode = EditorMode.Insert;
    }
    this.hiddenInput?.focus();
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
