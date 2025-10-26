import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Insert Mode', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('a command (append after cursor)', () => {
    it('should move cursor one position right and enter insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('a');
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
      expect(status.cursorX).toBe(1);
      expect(status.cursorY).toBe(0);
    });

    it('should NOT insert "a" character when entering insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('a');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(6);
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should work at end of line', async () => {
      editor.setContent(['hello']);
      editor.cursorX = 4;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('a');
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(5);
      expect(editor.content[0]).toBe('hello');
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should work on empty line', async () => {
      editor.setContent(['']);
      editor.cursorX = 0;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('a');
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(0);
      expect(editor.content[0]).toBe('');
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should allow typing after pressing a', async () => {
      editor.setContent(['test']);
      editor.cursorX = 1;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('a');
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(2);
      expect(editor.content[0]).toBe('test');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'x';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('texst');
      }
    });
  });

  describe('i command (insert before cursor)', () => {
    it('should enter insert mode without moving cursor', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 5;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('i');
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
      expect(status.cursorX).toBe(5);
      expect(status.cursorY).toBe(0);
    });

    it('should NOT insert "i" character when entering insert mode', async () => {
      editor.setContent(['hello world']);
      editor.cursorX = 3;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('i');
      await editor.updateComplete;
      
      expect(editor.content[0]).toBe('hello world');
      expect(editor.cursorX).toBe(3);
      const status = editor.getStatus();
      expect(status.mode).toBe('insert');
    });

    it('should allow typing after pressing i', async () => {
      editor.setContent(['test']);
      editor.cursorX = 2;
      editor.cursorY = 0;
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey('i');
      await editor.updateComplete;
      
      expect(editor.cursorX).toBe(2);
      expect(editor.content[0]).toBe('test');
      
      const hiddenInput = editor.shadowRoot?.querySelector('input');
      if (hiddenInput) {
        hiddenInput.value = 'x';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInput.dispatchEvent(inputEvent);
        await editor.updateComplete;
        
        expect(editor.content[0]).toBe('texst');
      }
    });
  });
});

