# VimEditor Tests

## Overview
This directory contains unit tests for the VimEditor Web Component using Vitest.

## Running Tests

```bash
# Run tests once
npm test

# Run tests in watch mode
npm run test

# Run tests with UI
npm run test:ui
```

## Test Structure

### Test Files
- `editing.test.ts` - Editing commands and text manipulation (77 tests)
- `navigation.test.ts` - Cursor movement commands (multiple tests)
- `visual-mode.test.ts` - Visual mode operations
- `insert-mode.test.ts` - Insert mode functionality
- `command-mode.test.ts` - Command mode operations
- `search-mode.test.ts` - Search functionality
- `multi-cursor.test.ts` - Multi-cursor operations
- `fast-jump.test.ts` - Fast jump navigation
- `text-objects.test.ts` - Text object selection
- `bracket-matching.test.ts` - Bracket matching
- `focus-management.test.ts` - Focus handling
- `system.test.ts` - System integration
- `clipboard.test.ts` - Clipboard operations (11 tests, **skipped** - see [CLIPBOARD_TESTS.md](./CLIPBOARD_TESTS.md))

### Test Coverage

#### 1. $ Key Navigation (Normal Mode)
- Tests cursor movement to end of line with `$` key
- Validates cursor position after key press
- Checks buffer state reflects cursor position

#### 2. Buffer System
- Validates buffer maintains correct content
- Verifies buffer reflects cursor position with correct colors
- Ensures buffer size matches render area

#### 3. Status System
- Tests `getStatus()` provides correct mode information
- Validates cursor position reporting
- Tracks mode changes (normal/insert/visual)

## Component Testing API

### Methods for Testing

```typescript
// Set content programmatically
editor.setContent(['line1', 'line2']);

// Get current editor status
const status = editor.getStatus();
// Returns: { mode, cursorX, cursorY, cursorVisible }

// Get buffer state
const buffer = editor.getBuffer();
// Returns: BufferCell[][]

// Manually update buffer
editor.updateBuffer();
```

### BufferCell Interface
```typescript
interface BufferCell {
  char: string;
  foreground: number[]; // [r, g, b]
  background: number[]; // [r, g, b]
}
```

### EditorStatus Interface
```typescript
interface EditorStatus {
  mode: 'normal' | 'insert' | 'visual';
  cursorX: number;
  cursorY: number;
  cursorVisible: boolean;
}
```

## Example Test

```typescript
it('should move cursor to end of line', () => {
  editor.setContent(['abc']);
  
  const event = new KeyboardEvent('keydown', { key: '$' });
  window.dispatchEvent(event);
  
  const status = editor.getStatus();
  expect(status.cursorX).toBe(2); // cursor on 'c'
  
  editor.updateBuffer();
  const buffer = editor.getBuffer();
  expect(buffer[0][2].char).toBe('c');
});
```

