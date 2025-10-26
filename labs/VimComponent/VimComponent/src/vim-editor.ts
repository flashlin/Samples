import { html, LitElement } from 'lit';
import { customElement, property, state } from 'lit/decorators.js';
import p5 from 'p5';
import exampleText from './example.txt?raw';
import { EditorMode, EditorStatus, BufferCell, EditorModeHandler, TextRange, IntellisenseItem, IntellisenseContext, VimEditorEventMap } from './vimEditorTypes';
import { ModeHandlerRegistry } from './handlers';
import { IntellisenseMenu } from './components/IntellisenseMenu';

/**
 * Vim-like editor component
 * 
 * @fires change - Fired when editor content changes
 * @fires keypress - Fired on every keypress
 * @fires vim-command - Fired when a command is executed in Command Mode
 * @fires intellisense - Fired to request intellisense suggestions
 */
@customElement('vim-editor')
export class VimEditor extends LitElement {
  declare addEventListener: <K extends keyof VimEditorEventMap>(
    type: K,
    listener: (this: VimEditor, ev: VimEditorEventMap[K]) => void,
    options?: boolean | AddEventListenerOptions
  ) => void;
  
  declare removeEventListener: <K extends keyof VimEditorEventMap>(
    type: K,
    listener: (this: VimEditor, ev: VimEditorEventMap[K]) => void,
    options?: boolean | EventListenerOptions
  ) => void;
  
  p5Instance: p5 | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private cursorBlinkInterval: number | null = null;
  private baseCharWidth = 9;
  private lineHeight = 20;
  private textPadding = 2;
  private textOffsetY = 5;
  private statusBarHeight = 24;
  hiddenInput: HTMLInputElement | null = null;
  private intellisenseMenu: IntellisenseMenu = new IntellisenseMenu();
  
  @state()
  private cursorVisible = true;
  
  @state()
  private hasFocus = false;
  
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

  @property({ type: String })
  width: string | number = 800;

  @property({ type: String })
  height: string | number = 600;

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
  
  visualStartX = 0;
  visualStartY = 0;
  
  // TVisual mode (multi-cursor visual based on tMarks)
  multiCursorClipboard: string[] = [];
  multiCursorOffsets: Array<{ offsetX: number; offsetY: number }> = [];
  
  private numberPrefix = '';
  
  private scrollOffsetX = 0;
  private scrollOffsetY = 0;

  fastJumpMatches: Array<{ x: number; y: number; label: string }> = [];
  fastJumpInput = '';
  previousMode: EditorMode.Normal | EditorMode.Visual | EditorMode.VisualLine = EditorMode.Normal;
  
  keyBuffer = '';
  
  searchKeyword = '';
  searchMatches: Array<{ y: number; x: number }> = [];
  currentMatchIndex = -1;
  searchHistory: Array<{ keyword: string; matches: Array<{ y: number; x: number }> }> = [];
  
  tMarks: Array<{ y: number; x: number }> = [];
  
  @property({ type: String })
  commandInput = '';
  
  @property({ type: String })
  searchInput = '';
  
  modeHandlerRegistry!: ModeHandlerRegistry;
  private currentModeHandler!: EditorModeHandler;
  private previousModeHandler!: EditorModeHandler;
  

  private history: Array<{ content: string[]; cursorX: number; cursorY: number }> = [];
  private historyIndex = -1;
  private maxHistorySize = 100;

  private parseSize(value: string | number, containerSize?: number): number {
    if (typeof value === 'number') {
      return value;
    }
    
    const str = value.toString().trim();
    
    if (str.endsWith('%')) {
      const percent = parseFloat(str);
      if (containerSize !== undefined) {
        return Math.floor((percent / 100) * containerSize);
      }
      return 800;
    }
    
    if (str.endsWith('px')) {
      return parseFloat(str);
    }
    
    const num = parseFloat(str);
    return isNaN(num) ? 800 : num;
  }

  private getComputedWidth(): number {
    const container = this.shadowRoot?.host as HTMLElement;
    // Use the actual rendered width of the host element
    if (container && container.clientWidth > 0) {
      return container.clientWidth;
    }
    const parentWidth = container?.parentElement?.clientWidth || window.innerWidth;
    return this.parseSize(this.width, parentWidth);
  }

  private getComputedHeight(): number {
    const container = this.shadowRoot?.host as HTMLElement;
    // Use the actual rendered height of the host element
    if (container && container.clientHeight > 0) {
      return container.clientHeight;
    }
    const parentHeight = container?.parentElement?.clientHeight || window.innerHeight;
    return this.parseSize(this.height, parentHeight);
  }

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


  findMatchesInVisibleRange(targetChar: string): Array<{ x: number; y: number; label: string }> {
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
    const computedWidth = this.getComputedWidth();
    const computedHeight = this.getComputedHeight();
    const editableWidth = Math.floor((computedWidth - 60) / this.baseCharWidth);
    const editableHeight = Math.floor((computedHeight - this.statusBarHeight) / this.lineHeight);
    
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
        
        const isCursor = contentY === this.cursorY && contentX === this.cursorX && this.cursorVisible && this.hasFocus;
        const isNormalMode = this.mode === EditorMode.Normal;
        const isSearchMode = this.mode === EditorMode.FastSearch;
        
        const isVisualSelection = this.mode === EditorMode.Visual && this.isInVisualSelection(contentY, contentX);
        const isVisualLineSelection = this.mode === EditorMode.VisualLine && this.isInVisualLineSelection(contentY);
        const isTVisualSelection = this.mode === EditorMode.TVisual && this.isInTVisualSelection(contentY, contentX);
        
        const isFastJumpMatch = this.mode === EditorMode.FastMatch && 
          this.fastJumpMatches.some(match => match.x === contentX && match.y === contentY);
        
        const isSearchMatch = (this.mode === EditorMode.FastSearch || this.mode === EditorMode.MultiInsert || this.mode === EditorMode.SearchInput || this.mode === EditorMode.Normal) && 
          this.isInSearchMatch(contentY, contentX);
        
        const isCurrentSearchMatch = (this.mode === EditorMode.FastSearch || this.mode === EditorMode.MultiInsert || this.mode === EditorMode.SearchInput || this.mode === EditorMode.Normal) && 
          this.currentMatchIndex >= 0 && 
          this.isInSearchMatch(contentY, contentX, this.currentMatchIndex);
        
        const isHighlighted = isVisualSelection || isVisualLineSelection || isTVisualSelection || isFastJumpMatch || isSearchMatch;
        
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

  private isInTVisualSelection(y: number, x: number): boolean {
    if (this.tMarks.length === 0 || this.multiCursorOffsets.length === 0) {
      return false;
    }
    
    // Check each tMark position with its independent offset
    for (let i = 0; i < this.tMarks.length; i++) {
      const mark = this.tMarks[i];
      const offset = this.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      
      const startX = mark.x;
      const startY = mark.y;
      const endX = mark.x + offset.offsetX;
      const endY = mark.y + offset.offsetY;
      
      const minY = Math.min(startY, endY);
      const maxY = Math.max(startY, endY);
      
      if (y < minY || y > maxY) {
        continue;
      }
      
      if (minY === maxY) {
        const minX = Math.min(startX, endX);
        const maxX = Math.max(startX, endX);
        if (x >= minX && x <= maxX) {
          return true;
        }
      } else {
        if (y === minY) {
          const minX = (startY === minY) ? startX : endX;
          if (x >= minX) {
            return true;
          }
        } else if (y === maxY) {
          const maxX = (startY === maxY) ? startX : endX;
          if (x <= maxX) {
            return true;
          }
        } else {
          return true;
        }
      }
    }
    
    return false;
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
    
    this.setupFocusManagement();
    this.createHiddenInput();
    this.waitForP5AndInitialize();
  }

  private setupFocusManagement() {
    const host = this.shadowRoot?.host as HTMLElement;
    if (host) {
      host.setAttribute('tabindex', '0');
      
      host.addEventListener('focus', () => {
        this.hasFocus = true;
        // If we're in a mode that needs hiddenInput, refocus it
        if (this.mode === EditorMode.Insert || 
            this.mode === EditorMode.MultiInsert || 
            this.mode === EditorMode.TInsert) {
          this.hiddenInput?.focus();
        }
        if (this.p5Instance) {
          this.p5Instance.redraw();
        }
      });
      
      host.addEventListener('blur', (event: FocusEvent) => {
        // Don't lose focus if the focus moved to hiddenInput
        if (event.relatedTarget === this.hiddenInput) {
          return;
        }
        this.hasFocus = false;
        this.cursorVisible = false;
        if (this.p5Instance) {
          this.p5Instance.redraw();
        }
      });
    }
  }

  private createHiddenInput() {
    this.hiddenInput = document.createElement('input');
    this.hiddenInput.setAttribute('type', 'text');
    this.hiddenInput.setAttribute('data-gramm', 'false');
    this.hiddenInput.setAttribute('data-gramm_editor', 'false');
    this.hiddenInput.setAttribute('data-enable-grammarly', 'false');
    this.hiddenInput.setAttribute('autocomplete', 'off');
    this.hiddenInput.setAttribute('autocorrect', 'off');
    this.hiddenInput.setAttribute('autocapitalize', 'off');
    this.hiddenInput.setAttribute('spellcheck', 'false');
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
      if (e.data && this.currentModeHandler.handleCompositionEnd) {
        this.currentModeHandler.handleCompositionEnd(this, e.data);
      }
      this.hiddenInput!.value = '';
    });
    
    this.hiddenInput.addEventListener('input', (e) => {
      if (!this.isComposing && this.hiddenInput) {
        const value = this.hiddenInput.value;
        if (value && this.currentModeHandler.handleInput) {
          this.currentModeHandler.handleInput(this, value);
        }
        this.hiddenInput.value = '';
      }
    });
    
    this.hiddenInput.focus();
    this.updateInputPosition();
  }

  updateInputPosition() {
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
    const width = this.getComputedWidth();
    const height = this.getComputedHeight();
    
    console.log('Canvas size:', width, height);

    const sketch = (p: p5) => {
      p.setup = () => {
        console.log('p5 setup called');
        const canvas = p.createCanvas(width, height);
        this.canvas = canvas.elt as HTMLCanvasElement;
        
        canvas.elt.style.cssText = `
          display: block !important;
          width: ${width}px !important;
          height: ${height}px !important;
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
        
        this.drawEditorBackground(p);
        this.drawLineNumbers(p);
        this.renderBuffer(p);
        this.drawTMarks(p);
        this.drawFastJumpLabels(p);
        this.drawStatusBar(p);
        this.drawBorder(p);
      };
    };

    // Create p5 instance directly using the imported p5 class
    this.p5Instance = new p5(sketch, this.shadowRoot as unknown as HTMLElement);

    this.cursorBlinkInterval = window.setInterval(() => {
      if (this.hasFocus) {
        this.cursorVisible = !this.cursorVisible;
        if (this.p5Instance) {
          this.p5Instance.redraw();
        }
      }
    }, 500);

    const host = this.shadowRoot?.host as HTMLElement;
    if (host) {
      host.addEventListener('keydown', this.handleKeyDown.bind(this));
    }
    
    if (this.canvas) {
      this.canvas.addEventListener('mousedown', () => {
        const host = this.shadowRoot?.host as HTMLElement;
        if (host) {
          host.focus();
        }
        this.hiddenInput?.focus();
      });
    }
  }

  private handleKeyDown(event: KeyboardEvent) {
    if (!this.hasFocus) {
      return;
    }
    
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
    
    this.emitKeyPress(key, event);
    
    if ((event.metaKey || event.ctrlKey) && key === 'v' && this.mode === EditorMode.Insert) {
      event.preventDefault();
      this.handlePaste();
      return;
    }
    
    if (event.ctrlKey && key === 'j' && this.mode === EditorMode.Insert) {
      event.preventDefault();
      console.log('[Intellisense] Ctrl+j pressed in Insert Mode');
      console.log('[Intellisense] Cursor position:', { x: this.cursorX, y: this.cursorY });
      console.log('[Intellisense] Current line:', this.content[this.cursorY] || '');
      const handler = this.currentModeHandler as any;
      if (handler.triggerIntellisense) {
        console.log('[Intellisense] Triggering intellisense handler');
        handler.triggerIntellisense(this);
      } else {
        console.warn('[Intellisense] No triggerIntellisense method found on handler');
      }
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

  handleMovement(key: string): boolean {
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




  moveCursorDown() {
    if (this.cursorY < this.content.length - 1) {
      this.cursorY += 1;
      this.adjustCursorX();
      this.updateInputPosition();
    }
  }

  moveCursorUp() {
    if (this.cursorY > 0) {
      this.cursorY -= 1;
      this.adjustCursorX();
      this.updateInputPosition();
    }
  }

  moveCursorLeft() {
    if (this.cursorX > 0) {
      this.cursorX -= 1;
      this.updateInputPosition();
    }
  }

  moveCursorRight() {
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

  moveToFirstLine() {
    this.cursorY = 0;
    this.moveCursorToLineStart();
  }

  private moveToLastLine() {
    this.cursorY = Math.max(0, this.content.length - 1);
    this.adjustCursorX();
    this.updateInputPosition();
  }

  jumpToMatchingBracket() {
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
    const computedWidth = this.getComputedWidth();
    const availablePixelWidth = (computedWidth - 60) - (this.baseCharWidth * 2);
    
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

  getInnerWordRange(): { startX: number; endX: number; y: number } | null {
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
  
  getInnerQuoteRange(quoteChar: string): TextRange | null {
    const range = this.findMultiLineQuoteRange(quoteChar);
    if (!range) {
      return null;
    }
    
    if (range.startY === range.endY && range.startX === range.endX) {
      return null;
    }
    
    return range;
  }
  
  getInnerBracketRange(): TextRange | null {
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
  
  findInnerBracketRange(openChar: string, closeChar: string): TextRange | null {
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
  
  private findMultiLineQuoteRange(quoteChar: string): TextRange | null {
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

  deleteWord() {
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

  deleteToWordEnd() {
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

  deleteToLineEnd() {
    const currentLine = this.content[this.cursorY] || '';
    if (this.cursorX >= currentLine.length) {
      return;
    }
    
    const beforeCursor = currentLine.substring(0, this.cursorX);
    this.content[this.cursorY] = beforeCursor;
    
    if (this.cursorX >= this.content[this.cursorY].length && this.content[this.cursorY].length > 0) {
      this.cursorX = this.content[this.cursorY].length - 1;
    }
  }

  deleteLine() {
    if (this.content.length === 1) {
      this.content = [''];
      this.cursorX = 0;
      return;
    }
    
    this.content.splice(this.cursorY, 1);
    
    if (this.cursorY >= this.content.length) {
      this.cursorY = this.content.length - 1;
    }
    
    this.cursorX = 0;
    this.adjustCursorX();
  }

  deleteLinesDown(count: number) {
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

  deleteLinesUp(count: number) {
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

  adjustCursorX() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0 && this.cursorX >= currentLine.length) {
      this.cursorX = currentLine.length - 1;
    }
  }

  adjustCursorForNormalMode() {
    const currentLine = this.content[this.cursorY] || '';
    if (currentLine.length > 0 && this.cursorX >= currentLine.length) {
      this.cursorX = currentLine.length - 1;
    }
  }

  insertLineBelow() {
    this.content.splice(this.cursorY + 1, 0, '');
    this.cursorY += 1;
    this.cursorX = 0;
    this.mode = EditorMode.Insert;
    this.updateInputPosition();
  }


  deleteMultiLineSelection(startY: number, endY: number, startX: number, endX: number) {
    const firstPart = this.content[startY].slice(0, startX);
    const lastPart = this.content[endY].slice(endX + 1);
    this.content[startY] = firstPart + lastPart;
    this.content.splice(startY + 1, endY - startY);
    this.cursorX = startX;
    this.cursorY = startY;
  }

  handleBackspace() {
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
    this.emitChange();
  }

  handleDelete() {
    const currentLine = this.content[this.cursorY];
    if (this.cursorX < currentLine.length) {
      this.content[this.cursorY] = currentLine.substring(0, this.cursorX) + currentLine.substring(this.cursorX + 1);
    } else if (this.cursorY < this.content.length - 1) {
      const nextLine = this.content[this.cursorY + 1];
      this.content[this.cursorY] = currentLine + nextLine;
      this.content.splice(this.cursorY + 1, 1);
    }
    this.updateInputPosition();
    this.emitChange();
  }

  handleEnter() {
    const currentLine = this.content[this.cursorY];
    const lineBeforeCursor = currentLine.substring(0, this.cursorX);
    const lineAfterCursor = currentLine.substring(this.cursorX);
    
    this.content[this.cursorY] = lineBeforeCursor;
    this.content.splice(this.cursorY + 1, 0, lineAfterCursor);
    
    this.cursorY += 1;
    this.cursorX = 0;
    this.updateInputPosition();
    this.emitChange();
  }

  insertCharacter(char: string) {
    const currentLine = this.content[this.cursorY];
    this.content[this.cursorY] = 
      currentLine.substring(0, this.cursorX) + 
      char + 
      currentLine.substring(this.cursorX);
    this.cursorX += 1;
    this.updateInputPosition();
    this.emitChange();
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

  async copyToClipboard(content: string, isLinewise: boolean = false): Promise<void> {
    const textToCopy = isLinewise ? content + '\n' : content;
    await navigator.clipboard.writeText(textToCopy);
  }

  async pasteAfterCursor() {
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

  async pasteBeforeCursor() {
    try {
      const text = await navigator.clipboard.readText();
      this.pasteTextBeforeCursor(text);
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
    
    // Check if this is a line-wise paste (ends with newline)
    const isLinewise = text.endsWith('\n');
    
    if (isLinewise) {
      // Line-wise paste: insert complete lines after current line
      const lines = text.slice(0, -1).split('\n'); // Remove trailing newline before split
      
      // Insert all lines after the current line
      this.content.splice(this.cursorY + 1, 0, ...lines);
      
      // Move cursor to the first inserted line
      this.cursorY += 1;
      this.cursorX = 0;
    } else {
      // Character-wise paste: insert at cursor position
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
    }
    
    this.updateInputPosition();
  }

  private pasteTextBeforeCursor(text: string) {
    const currentLine = this.content[this.cursorY] || '';
    
    // Check if this is a line-wise paste (ends with newline)
    const isLinewise = text.endsWith('\n');
    
    if (isLinewise) {
      // Line-wise paste: insert complete lines before current line
      const lines = text.slice(0, -1).split('\n'); // Remove trailing newline before split
      
      // Insert all lines before the current line
      this.content.splice(this.cursorY, 0, ...lines);
      
      // Cursor stays at the beginning of the first inserted line
      this.cursorX = 0;
    } else {
      // Character-wise paste: insert at cursor position
      const lines = text.split('\n');
      
      if (lines.length === 1) {
        const insertPosition = this.cursorX;
        const beforeInsert = currentLine.substring(0, insertPosition);
        const afterInsert = currentLine.substring(insertPosition);
        
        this.content[this.cursorY] = beforeInsert + text + afterInsert;
        this.cursorX = insertPosition + text.length - 1;
      } else {
        const insertPosition = this.cursorX;
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
    }
    
    this.updateInputPosition();
  }

  saveHistory(cursorPos: { cursorX: number; cursorY: number } | null = null) {
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

  undo() {
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

  showIntellisense(items: IntellisenseItem[]): void {
    if (!this.canvas) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const cursorX = 40 + this.cursorX * this.baseCharWidth;
    const cursorY = this.cursorY * this.lineHeight;
    
    const absoluteX = rect.left + cursorX;
    const absoluteY = rect.top + cursorY;
    
    this.intellisenseMenu.show(items, absoluteX, absoluteY, document.body);
  }

  hideIntellisense(): void {
    this.intellisenseMenu.hide();
  }

  replaceWordAtCursor(oldWord: string, newWord: string): void {
    const currentLine = this.content[this.cursorY];
    if (!currentLine) return;
    
    const cursorPos = this.cursorX;
    const wordStart = cursorPos - oldWord.length;
    if (wordStart < 0) return;
    
    const wordAtCursor = currentLine.substring(wordStart, cursorPos);
    if (wordAtCursor !== oldWord) return;
    
    this.content[this.cursorY] = 
      currentLine.substring(0, wordStart) + 
      newWord + 
      currentLine.substring(cursorPos);
    
    this.cursorX = wordStart + newWord.length;
    
    if (this.p5Instance) {
      this.p5Instance.redraw();
    }
    
    this.emitChange();
  }

  emitChange(): void {
    const event = new CustomEvent('change', {
      detail: { content: [...this.content] }
    });
    this.dispatchEvent(event);
  }

  emitKeyPress(key: string, originalEvent: KeyboardEvent): void {
    const event = new CustomEvent('keypress', {
      detail: {
        key,
        mode: this.mode,
        ctrlKey: originalEvent.ctrlKey,
        shiftKey: originalEvent.shiftKey,
        altKey: originalEvent.altKey,
        metaKey: originalEvent.metaKey,
        cursorX: this.cursorX,
        cursorY: this.cursorY
      }
    });
    this.dispatchEvent(event);
  }
  
  emitCommand(command: string): void {
    const event = new CustomEvent('vim-command', {
      detail: { command },
      bubbles: true,
      composed: true
    });
    this.dispatchEvent(event);
  }
  
  emitIntellisense(context: IntellisenseContext): void {
    const event = new CustomEvent('intellisense', {
      detail: context
    });
    this.dispatchEvent(event);
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
    const host = this.shadowRoot?.host as HTMLElement;
    if (host) {
      host.removeEventListener('keydown', this.handleKeyDown.bind(this));
    }
    super.disconnectedCallback();
  }

  private drawBorder(p: p5) {
    p.noFill();
    if (this.hasFocus) {
      p.stroke(100, 149, 237);
      p.strokeWeight(2);
      // When strokeWeight is 2, draw from (1,1) to avoid clipping
      p.rect(1, 1, p.width - 2, p.height - 2);
    } else {
      p.stroke(100);
      p.strokeWeight(1);
      // When strokeWeight is 1, draw from (0.5,0.5) for pixel-perfect rendering
      p.rect(0.5, 0.5, p.width - 1, p.height - 1);
    }
    p.noStroke();
    p.strokeWeight(1);
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
    if ((this.mode !== EditorMode.Insert && this.mode !== EditorMode.MultiInsert && this.mode !== EditorMode.TInsert) || !this.cursorVisible || !this.hasFocus) {
      return;
    }
    
    if (this.mode === EditorMode.TInsert && this.tMarks.length > 0) {
      const currentMarkIndex = this.findCurrentTMarkIndex();
      if (currentMarkIndex !== -1) {
        const currentMark = this.tMarks[currentMarkIndex];
        const offsetInMark = this.cursorX - currentMark.x;
        
        for (const mark of this.tMarks) {
          this.drawCursorAtPosition(p, mark.x + offsetInMark, mark.y);
        }
      }
    } else {
      this.drawCursorAtPosition(p, this.cursorX, this.cursorY);
    }
  }
  
  private findCurrentTMarkIndex(): number {
    if (this.tMarks.length === 0) return -1;
    
    for (let i = 0; i < this.tMarks.length; i++) {
      const mark = this.tMarks[i];
      if (mark.y === this.cursorY && mark.x === this.cursorX) {
        return i;
      }
    }
    
    for (let i = this.tMarks.length - 1; i >= 0; i--) {
      const mark = this.tMarks[i];
      if (mark.y < this.cursorY || (mark.y === this.cursorY && mark.x <= this.cursorX)) {
        return i;
      }
    }
    
    return 0;
  }
  
  private drawCursorAtPosition(p: p5, x: number, y: number) {
    const bufferY = y - this.scrollOffsetY;
    const bufferX = x - this.scrollOffsetX;
    
    if (bufferY < 0 || bufferY >= this.bufferHeight || bufferX < 0 || bufferX >= this.bufferWidth) {
      return;
    }
    
    const line = this.content[y] || '';
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
    const editorHeight = p.height - this.statusBarHeight;
    const statusY = editorHeight;
    
    p.fill(50);
    p.rect(0, statusY, p.width, this.statusBarHeight);
    
    p.fill(255);
    
    if (this.mode === EditorMode.Command) {
      const commandText = this.commandInput;
      p.text(commandText, 10, statusY + 3);
      
      if (this.cursorVisible) {
        const textWidth = p.textWidth(commandText);
        p.fill(255);
        p.rect(10 + textWidth, statusY + 2, 2, this.statusBarHeight - 4);
      }
    } else if (this.mode === EditorMode.SearchInput) {
      const searchText = this.searchInput;
      p.text(searchText, 10, statusY + 3);
      
      if (this.cursorVisible) {
        const textWidth = p.textWidth(searchText);
        p.fill(255);
        p.rect(10 + textWidth, statusY + 2, 2, this.statusBarHeight - 4);
      }
      
      if (this.searchMatches.length > 0) {
        const matchInfo = ` (${this.searchMatches.length} matches)`;
        const searchTextWidth = p.textWidth(searchText);
        p.text(matchInfo, 10 + searchTextWidth + 10, statusY + 3);
      }
    } else {
      let statusText = `Mode: ${this.mode} | Line: ${this.cursorY + 1}, Col: ${this.getDisplayColumn() + 1}`;
      if (this.lastKeyPressed) {
        statusText += ` | Key: "${this.lastKeyPressed}"`;
      }
      if (this.searchMatches.length > 0) {
        statusText += ` | Matches: ${this.searchMatches.length}`;
      }
      p.text(statusText, 10, statusY + 3);
    }
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


  enterInsertMode() {
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
    const widthStr = typeof this.width === 'number' ? `${this.width}px` : this.width;
    const heightStr = typeof this.height === 'number' ? `${this.height}px` : this.height;
    
    return html`
      <style>
        :host {
          display: block;
          width: ${widthStr};
          height: ${heightStr};
          background-color: #000;
          position: relative;
        }
      </style>
    `;
  }
}
