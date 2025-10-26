import { EditorMode, EditorModeHandler } from '../vimEditorTypes';
import { NormalModeHandler } from './NormalModeHandler';
import { InsertModeHandler } from './InsertModeHandler';
import { VisualModeHandler } from './VisualModeHandler';
import { VisualLineModeHandler } from './VisualLineModeHandler';
import { FastJumpModeHandler } from './FastJumpModeHandler';
import { FastMatchModeHandler } from './FastMatchModeHandler';
import { FastSearchModeHandler } from './FastSearchModeHandler';
import { MultiInsertModeHandler } from './MultiInsertModeHandler';
import { TInsertModeHandler } from './TInsertModeHandler';
import { TVisualModeHandler } from './TVisualModeHandler';
import { CommandModeHandler } from './CommandModeHandler';

export class ModeHandlerRegistry {
  private handlers: Map<EditorMode, EditorModeHandler> = new Map();
  
  constructor() {
    this.registerHandler(new NormalModeHandler());
    this.registerHandler(new InsertModeHandler());
    this.registerHandler(new VisualModeHandler());
    this.registerHandler(new VisualLineModeHandler());
    this.registerHandler(new FastJumpModeHandler());
    this.registerHandler(new FastMatchModeHandler());
    this.registerHandler(new FastSearchModeHandler());
    this.registerHandler(new MultiInsertModeHandler());
    this.registerHandler(new TInsertModeHandler());
    this.registerHandler(new TVisualModeHandler());
    this.registerHandler(new CommandModeHandler());
  }
  
  private registerHandler(handler: EditorModeHandler): void {
    this.handlers.set(handler.mode, handler);
  }
  
  getHandler(mode: EditorMode): EditorModeHandler {
    const handler = this.handlers.get(mode);
    if (!handler) {
      throw new Error(`No handler found for mode: ${mode}`);
    }
    return handler;
  }
}

