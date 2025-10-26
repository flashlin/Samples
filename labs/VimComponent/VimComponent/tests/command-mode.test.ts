import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { setupP5Mock, createTestEditor, cleanupTestEditor, pressKey } from './test-helpers';

setupP5Mock();

import '../src/vim-editor';

describe('VimEditor - Command Mode', () => {
  let editor: any;

  beforeEach(async () => {
    editor = await createTestEditor();
  });

  afterEach(() => {
    cleanupTestEditor(editor);
  });

  describe('entering command mode', () => {
    it('should enter command mode when pressing ":"', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('command');
    });

    it('should initialize commandInput with ":"', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':');
    });
  });

  describe('input handling', () => {
    it('should accumulate characters in commandInput', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'w';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':w');
      
      editor.hiddenInput.value = 'q';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':wq');
    });

    it('should support Chinese input', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = '中文';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':中文');
    });

    it('should handle composition events', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.currentModeHandler.handleCompositionEnd(editor, '測試');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':測試');
    });
  });

  describe('backspace handling', () => {
    it('should delete last character on Backspace', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'set';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':set');
      
      pressKey('Backspace');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':se');
    });

    it('should not delete ":" prefix', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':');
      
      pressKey('Backspace');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':');
    });

    it('should handle multiple backspaces', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'abc';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':abc');
      
      pressKey('Backspace');
      await editor.updateComplete;
      expect(editor.commandInput).toBe(':ab');
      
      pressKey('Backspace');
      await editor.updateComplete;
      expect(editor.commandInput).toBe(':a');
      
      pressKey('Backspace');
      await editor.updateComplete;
      expect(editor.commandInput).toBe(':');
      
      pressKey('Backspace');
      await editor.updateComplete;
      expect(editor.commandInput).toBe(':');
    });
  });

  describe('Enter key handling', () => {
    it('should dispatch vim-command event on Enter', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'set number';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(eventSpy).toHaveBeenCalledTimes(1);
      expect(eventSpy.mock.calls[0][0].detail.command).toBe('set number');
    });

    it('should return to normal mode after Enter', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'w';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should clear commandInput after Enter', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'test';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe('');
    });

    it('should not include ":" prefix in command event', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'wq';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(eventSpy.mock.calls[0][0].detail.command).toBe('wq');
      expect(eventSpy.mock.calls[0][0].detail.command).not.toContain(':');
    });

    it('should send empty command if only ":" is present', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(eventSpy).toHaveBeenCalledTimes(1);
      expect(eventSpy.mock.calls[0][0].detail.command).toBe('');
    });
  });

  describe('Escape key handling', () => {
    it('should return to normal mode on Escape', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      expect(editor.mode).toBe('command');
      
      pressKey('Escape');
      await editor.updateComplete;
      
      const status = editor.getStatus();
      expect(status.mode).toBe('normal');
    });

    it('should not dispatch vim-command event on Escape', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'test command';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Escape');
      await editor.updateComplete;
      
      expect(eventSpy).not.toHaveBeenCalled();
    });

    it('should clear commandInput on Escape', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'some command';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe(':some command');
      
      pressKey('Escape');
      await editor.updateComplete;
      
      expect(editor.commandInput).toBe('');
    });
  });

  describe('event details', () => {
    it('should include correct command in event detail', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'echo hello';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      const event = eventSpy.mock.calls[0][0];
      expect(event.detail).toHaveProperty('command');
      expect(event.detail.command).toBe('echo hello');
    });

    it('should have bubbles and composed properties', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'test';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      const event = eventSpy.mock.calls[0][0];
      expect(event.bubbles).toBe(true);
      expect(event.composed).toBe(true);
    });
  });

  describe('complex command scenarios', () => {
    it('should handle commands with spaces', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'set tabstop=4 shiftwidth=4';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(eventSpy.mock.calls[0][0].detail.command).toBe('set tabstop=4 shiftwidth=4');
    });

    it('should handle commands with special characters', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 's/old/new/g';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(eventSpy.mock.calls[0][0].detail.command).toBe('s/old/new/g');
    });

    it('should handle mixed English and Chinese commands', async () => {
      editor.setContent(['hello world']);
      editor.mode = 'normal';
      await editor.updateComplete;
      
      const eventSpy = vi.fn();
      editor.addEventListener('vim-command', eventSpy);
      
      pressKey(':');
      await editor.updateComplete;
      
      editor.hiddenInput.value = 'echo 你好世界';
      editor.hiddenInput.dispatchEvent(new Event('input'));
      await editor.updateComplete;
      
      pressKey('Enter');
      await editor.updateComplete;
      
      expect(eventSpy.mock.calls[0][0].detail.command).toBe('echo 你好世界');
    });
  });
});

