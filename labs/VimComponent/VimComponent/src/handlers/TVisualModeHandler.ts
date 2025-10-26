import { EditorMode, BaseModeHandler, IVimEditor } from '../vimEditorTypes';

export class TVisualModeHandler extends BaseModeHandler {
  readonly mode = EditorMode.TVisual;
  
  shouldPreventDefault(key: string): boolean {
    return true;
  }
  
  onEnter(editor: IVimEditor): void {
    // Initialize independent offsets for each tMark
    editor.multiCursorOffsets = editor.tMarks.map(() => ({ offsetX: 0, offsetY: 0 }));
  }
  
  onExit(editor: IVimEditor): void {
    // Clean up offsets when exiting
    editor.multiCursorOffsets = [];
  }
  
  async handleKey(key: string, editor: IVimEditor): Promise<void> {
    // Handle independent movement for each tMark
    if (this.handleIndependentMovement(key, editor)) {
      return;
    }
    
    if (key === 'Escape') {
      editor.mode = EditorMode.Normal;
      return;
    }
    
    switch (key) {
      case 'y':
        await this.yankTVisualSelections(editor);
        editor.mode = EditorMode.Normal;
        break;
      case 'd':
      case 'x':
        // TODO: Implement delete for TVisual mode if needed
        editor.mode = EditorMode.Normal;
        break;
    }
  }
  
  private handleIndependentMovement(key: string, editor: IVimEditor): boolean {
    switch (key) {
      case 'e':
        this.moveToWordEndAtEachMark(editor);
        return true;
      case 'w':
        this.moveToNextWordAtEachMark(editor);
        return true;
      case 'b':
        this.moveToPreviousWordAtEachMark(editor);
        return true;
      case '$':
        this.moveToLineEndAtEachMark(editor);
        return true;
      case '^':
        this.moveToLineStartAtEachMark(editor);
        return true;
      case 'h':
      case 'ArrowLeft':
        this.moveLeftAtEachMark(editor);
        return true;
      case 'l':
      case 'ArrowRight':
        this.moveRightAtEachMark(editor);
        return true;
      default:
        return false;
    }
  }
  
  private moveToWordEndAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const currentOffset = editor.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      
      const currentX = mark.x + currentOffset.offsetX;
      const currentY = mark.y + currentOffset.offsetY;
      const currentLine = editor.content[currentY] || '';
      
      if (currentX >= currentLine.length) {
        continue;
      }
      
      const currentChar = currentLine[currentX];
      const currentType = this.getCharType(currentChar);
      
      if (currentType === 'space') {
        continue;
      }
      
      let endX = currentX;
      const nextChar = currentLine[currentX + 1];
      
      if (nextChar && this.getCharType(nextChar) !== currentType) {
        endX = currentX + 1;
        while (endX < currentLine.length && this.getCharType(currentLine[endX]) === 'space') {
          endX++;
        }
        if (endX >= currentLine.length) {
          continue;
        }
      }
      
      const newType = this.getCharType(currentLine[endX]);
      
      for (let j = endX + 1; j < currentLine.length; j++) {
        const charType = this.getCharType(currentLine[j]);
        if (charType !== newType) {
          endX = j - 1;
          break;
        }
        if (j === currentLine.length - 1) {
          endX = j;
        }
      }
      
      editor.multiCursorOffsets[i] = {
        offsetX: endX - mark.x,
        offsetY: 0
      };
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private moveToNextWordAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const currentOffset = editor.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      
      const currentX = mark.x + currentOffset.offsetX;
      const currentY = mark.y + currentOffset.offsetY;
      const currentLine = editor.content[currentY] || '';
      
      let newX = currentX + 1;
      
      if (newX < currentLine.length) {
        const currentType = this.getCharType(currentLine[currentX]);
        
        while (newX < currentLine.length && this.getCharType(currentLine[newX]) === currentType) {
          newX++;
        }
        
        while (newX < currentLine.length && this.getCharType(currentLine[newX]) === 'space') {
          newX++;
        }
        
        if (newX < currentLine.length) {
          editor.multiCursorOffsets[i] = {
            offsetX: newX - mark.x,
            offsetY: 0
          };
        }
      }
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private moveToPreviousWordAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const currentOffset = editor.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      
      const currentX = mark.x + currentOffset.offsetX;
      const currentY = mark.y + currentOffset.offsetY;
      const currentLine = editor.content[currentY] || '';
      
      if (currentX <= 0) continue;
      
      let newX = currentX - 1;
      
      while (newX >= 0 && this.getCharType(currentLine[newX]) === 'space') {
        newX--;
      }
      
      if (newX >= 0) {
        const charType = this.getCharType(currentLine[newX]);
        
        while (newX > 0 && this.getCharType(currentLine[newX - 1]) === charType) {
          newX--;
        }
        
        editor.multiCursorOffsets[i] = {
          offsetX: newX - mark.x,
          offsetY: 0
        };
      }
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private moveToLineEndAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const line = editor.content[mark.y] || '';
      
      if (line.length > 0) {
        editor.multiCursorOffsets[i] = {
          offsetX: line.length - 1 - mark.x,
          offsetY: 0
        };
      }
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private moveToLineStartAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const line = editor.content[mark.y] || '';
      
      let firstNonSpace = 0;
      for (let j = 0; j < line.length; j++) {
        if (line[j] !== ' ' && line[j] !== '\t') {
          firstNonSpace = j;
          break;
        }
      }
      
      editor.multiCursorOffsets[i] = {
        offsetX: firstNonSpace - mark.x,
        offsetY: 0
      };
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private moveLeftAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const currentOffset = editor.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      if (currentOffset.offsetX > 0) {
        editor.multiCursorOffsets[i] = {
          offsetX: currentOffset.offsetX - 1,
          offsetY: currentOffset.offsetY
        };
      }
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private moveRightAtEachMark(editor: IVimEditor): void {
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const currentOffset = editor.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      const line = editor.content[mark.y] || '';
      
      if (mark.x + currentOffset.offsetX < line.length - 1) {
        editor.multiCursorOffsets[i] = {
          offsetX: currentOffset.offsetX + 1,
          offsetY: currentOffset.offsetY
        };
      }
    }
    
    if (editor.p5Instance) {
      editor.p5Instance.redraw();
    }
  }
  
  private getCharType(char: string): 'word' | 'chinese' | 'space' {
    if (/[\u4e00-\u9fa5]/.test(char)) {
      return 'chinese';
    } else if (/\w/.test(char)) {
      return 'word';
    } else {
      return 'space';
    }
  }
  
  private async yankTVisualSelections(editor: IVimEditor): Promise<void> {
    const selections: string[] = [];
    
    for (let i = 0; i < editor.tMarks.length; i++) {
      const mark = editor.tMarks[i];
      const offset = editor.multiCursorOffsets[i] || { offsetX: 0, offsetY: 0 };
      
      const startX = mark.x;
      const startY = mark.y;
      const endX = mark.x + offset.offsetX;
      const endY = mark.y + offset.offsetY;
      
      const minY = Math.min(startY, endY);
      const maxY = Math.max(startY, endY);
      
      if (minY === maxY) {
        const minX = Math.min(startX, endX);
        const maxX = Math.max(startX, endX);
        const line = editor.content[minY] || '';
        selections.push(line.slice(minX, maxX + 1));
      } else {
        let result = '';
        for (let y = minY; y <= maxY; y++) {
          const line = editor.content[y] || '';
          if (y === minY) {
            const minX = (startY === minY) ? startX : endX;
            result += line.slice(minX) + '\n';
          } else if (y === maxY) {
            const maxX = (startY === maxY) ? startX : endX;
            result += line.slice(0, maxX + 1);
          } else {
            result += line + '\n';
          }
        }
        selections.push(result);
      }
    }
    
    editor.multiCursorClipboard = selections;
    
    if (selections.length > 0) {
      await editor.copyToClipboard(selections.join('\n'), false);
    }
  }
}

