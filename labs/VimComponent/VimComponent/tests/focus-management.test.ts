import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey, pressKeys } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Focus Management', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('multi-instance focus management', () => {
    let editorA: any;
    let editorB: any;

    beforeEach(async () => {
      // Create two editor instances
      editorA = document.createElement('vim-editor');
      editorB = document.createElement('vim-editor');
      document.body.appendChild(editorA);
      document.body.appendChild(editorB);
      
      await new Promise(resolve => setTimeout(resolve, 50));
      
      editorA.setContent(['Editor A line 1', 'Editor A line 2']);
      editorB.setContent(['Editor B line 1', 'Editor B line 2']);
      await editorA.updateComplete;
      await editorB.updateComplete;
    });

    afterEach(() => {
      if (editorA && editorA.parentNode) {
        editorA.parentNode.removeChild(editorA);
      }
      if (editorB && editorB.parentNode) {
        editorB.parentNode.removeChild(editorB);
      }
    });

    it('should maintain insert mode functionality after focus switch', async () => {
      // Editor A enters insert mode
      editorA.mode = 'normal';
      editorA.cursorX = 8;
      editorA.cursorY = 0;
      editorA.hasFocus = true;
      editorA.focus();
      await editorA.updateComplete;
      
      // Press 'i' to enter insert mode
      const iEvent = new KeyboardEvent('keydown', { key: 'i', bubbles: true });
      editorA.dispatchEvent(iEvent);
      await editorA.updateComplete;
      
      expect(editorA.mode).toBe('insert');
      
      // Simulate input in editor A
      const hiddenInputA = editorA.shadowRoot?.querySelector('input');
      if (hiddenInputA) {
        hiddenInputA.value = 'X';
        const inputEvent = new Event('input', { bubbles: true });
        hiddenInputA.dispatchEvent(inputEvent);
        await editorA.updateComplete;
        
        expect(editorA.content[0]).toBe('Editor AX line 1');
      }
      
      // Switch focus to editor B
      editorA.hasFocus = false;
      editorB.hasFocus = true;
      editorB.focus();
      await editorB.updateComplete;
      
      // Editor A should still be in insert mode but without focus
      expect(editorA.mode).toBe('insert');
      expect(editorA.hasFocus).toBe(false);
      expect(editorB.hasFocus).toBe(true);
      
      // Switch focus back to editor A
      editorB.hasFocus = false;
      editorA.hasFocus = true;
      
      // Simulate the focus event to trigger the refocus logic
      const focusEvent = new FocusEvent('focus', { bubbles: true });
      const hostA = editorA.shadowRoot?.host as HTMLElement;
      hostA.dispatchEvent(focusEvent);
      await editorA.updateComplete;
      
      // Editor A should still be in insert mode and accept input
      expect(editorA.mode).toBe('insert');
      expect(editorA.hasFocus).toBe(true);
      
      // Test input still works
      if (hiddenInputA) {
        hiddenInputA.value = 'Y';
        const inputEvent2 = new Event('input', { bubbles: true });
        hiddenInputA.dispatchEvent(inputEvent2);
        await editorA.updateComplete;
        
        expect(editorA.content[0]).toBe('Editor AXY line 1');
      }
    });

    it('should handle multi-insert mode after focus switch', async () => {
      // Editor A enters multi-insert mode
      editorA.mode = 'normal';
      editorA.cursorX = 0;
      editorA.cursorY = 0;
      editorA.hasFocus = true;
      editorA.focus();
      
      // Simulate entering multi-insert mode
      editorA.mode = 'multi-insert';
      editorA.searchKeyword = 'A';
      editorA.searchMatches = [
        { y: 0, x: 7 },
        { y: 1, x: 7 }
      ];
      editorA.currentMatchIndex = 0;
      await editorA.updateComplete;
      
      // Switch focus to editor B and back
      editorA.hasFocus = false;
      editorB.hasFocus = true;
      await editorB.updateComplete;
      
      editorB.hasFocus = false;
      editorA.hasFocus = true;
      
      // Trigger focus event
      const focusEvent = new FocusEvent('focus', { bubbles: true });
      const hostA = editorA.shadowRoot?.host as HTMLElement;
      hostA.dispatchEvent(focusEvent);
      await editorA.updateComplete;
      
      // Should still be in multi-insert mode and functional
      expect(editorA.mode).toBe('multi-insert');
      expect(editorA.hasFocus).toBe(true);
    });

    it('should handle t-insert mode after focus switch', async () => {
      // Editor A enters t-insert mode
      editorA.mode = 'normal';
      editorA.cursorX = 5;
      editorA.cursorY = 0;
      editorA.hasFocus = true;
      editorA.focus();
      
      // Add t-marks
      editorA.tMarks = [
        { x: 5, y: 0 },
        { x: 5, y: 1 }
      ];
      
      // Enter t-insert mode
      editorA.mode = 't-insert';
      await editorA.updateComplete;
      
      // Switch focus to editor B and back
      editorA.hasFocus = false;
      editorB.hasFocus = true;
      await editorB.updateComplete;
      
      editorB.hasFocus = false;
      editorA.hasFocus = true;
      
      // Trigger focus event
      const focusEvent = new FocusEvent('focus', { bubbles: true });
      const hostA = editorA.shadowRoot?.host as HTMLElement;
      hostA.dispatchEvent(focusEvent);
      await editorA.updateComplete;
      
      // Should still be in t-insert mode and functional
      expect(editorA.mode).toBe('t-insert');
      expect(editorA.hasFocus).toBe(true);
    });

    it('should not refocus hiddenInput in normal mode', async () => {
      // Editor A in normal mode
      editorA.mode = 'normal';
      editorA.cursorX = 0;
      editorA.cursorY = 0;
      editorA.hasFocus = true;
      editorA.focus();
      await editorA.updateComplete;
      
      const hiddenInputA = editorA.shadowRoot?.querySelector('input');
      const initialFocus = document.activeElement;
      
      // Switch focus to editor B and back
      editorA.hasFocus = false;
      editorB.hasFocus = true;
      await editorB.updateComplete;
      
      editorB.hasFocus = false;
      editorA.hasFocus = true;
      
      // Trigger focus event
      const focusEvent = new FocusEvent('focus', { bubbles: true });
      const hostA = editorA.shadowRoot?.host as HTMLElement;
      hostA.dispatchEvent(focusEvent);
      await editorA.updateComplete;
      
      // In normal mode, hiddenInput should NOT automatically get focus
      expect(editorA.mode).toBe('normal');
      expect(editorA.hasFocus).toBe(true);
      // hiddenInput should not be the active element in normal mode
      expect(document.activeElement).not.toBe(hiddenInputA);
    });
  });
});
