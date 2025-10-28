# VimComponent

A Vim-like editor component built with Lit and p5.js.

## Installation

```bash
npm install vimcomponent
# or
pnpm add vimcomponent
# or
yarn add vimcomponent
```

## Peer Dependencies

This component requires the following peer dependencies to be installed:

- **lit** ^3.0.0
- **p5** ^1.6.0

Install them with:

```bash
pnpm add lit p5@^1.6.0
```

> **Important**: Make sure to install `p5@^1.6.0` (version 1.x), not the latest 2.x version, as the component is built for p5.js v1.

## Usage

### In HTML

```html
<!DOCTYPE html>
<html>
<head>
  <script type="module">
    import 'vimcomponent';
  </script>
</head>
<body>
  <vim-editor 
    width="800" 
    height="400"
    content="// Your code here"
  ></vim-editor>
</body>
</html>
```

### In Vue 3

```vue
<template>
  <vim-editor
    :width="editorWidth"
    :height="editorHeight"
    :content="editorContent"
    @change="handleChange"
  />
</template>

<script setup>
import { ref } from 'vue'
import 'vimcomponent'

const editorWidth = '90%'
const editorHeight = '300px'
const editorContent = ref('// Enter your code here')

const handleChange = (event) => {
  editorContent.value = event.detail.content
}
</script>
```

### In React

```jsx
import { useEffect, useRef } from 'react';
import 'vimcomponent';

function App() {
  const editorRef = useRef(null);

  useEffect(() => {
    const editor = editorRef.current;
    
    const handleChange = (e) => {
      console.log('Content changed:', e.detail.content);
    };
    
    editor?.addEventListener('change', handleChange);
    return () => editor?.removeEventListener('change', handleChange);
  }, []);

  return (
    <vim-editor
      ref={editorRef}
      width="90%"
      height="300px"
      content="// Your code here"
    />
  );
}
```

## Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `width` | string | '100%' | Width of the editor |
| `height` | string | '400px' | Height of the editor |
| `content` | string | '' | Initial content of the editor |

## Events

| Event | Detail | Description |
|-------|--------|-------------|
| `change` | `{ content: string }` | Fired when the editor content changes |

## Vim Features

The editor supports basic Vim keybindings and modes:

- **Normal mode**: Navigate and manipulate text
- **Insert mode**: Edit text
- **Visual mode**: Select text

### Common Commands

- `i` - Enter insert mode
- `Esc` - Return to normal mode
- `h, j, k, l` - Move cursor (left, down, up, right)
- `w, b` - Move by word (forward, backward)
- `0, $` - Move to start/end of line
- `gg, G` - Move to start/end of file
- `dd` - Delete line
- `D` - Delete from cursor to end of line
- `yy` - Yank (copy) line
- `p` - Paste
- `u` - Undo
- `Ctrl+r` - Redo

## Development

```bash
# Install dependencies
pnpm install

# Run development server
pnpm run dev

# Build for production
pnpm run build

# Run tests
pnpm test
```

## Troubleshooting

### Issue: "Waiting for p5.js to load..." keeps showing

If you see this message continuously, it means p5.js is not being detected correctly. This issue has been fixed in the current version.

**Solution:**
1. Make sure you have p5.js installed: `pnpm add 'p5@^1.6.0'`
2. Rebuild VimComponent: `pnpm run build`
3. Restart your development server

**Root cause:** 
When using ES module imports (`import p5 from 'p5'`), p5 is not automatically mounted to `window.p5`. The component now checks for both the module import and ensures window.p5 is available.

For more details, see [TROUBLESHOOTING.md](../TROUBLESHOOTING.md).

## Architecture

### ES Module Approach

VimComponent uses a **pure ES module** approach for p5.js integration:

```typescript
import p5 from 'p5';

// Create p5 instance programmatically
this.p5Instance = new p5(sketch, this.shadowRoot);
```

**Benefits:**
- ✅ No global namespace pollution
- ✅ Full TypeScript type support
- ✅ Tree-shaking support for smaller bundles
- ✅ Module-based dependency management
- ✅ Can create multiple isolated instances

For more details, see [ES_MODULE_GUIDE.md](../ES_MODULE_GUIDE.md).

## Version History

### v1.0.0+
- ✅ Fixed: p5.js ES module import detection
- ✅ Improved: Pure ES module approach (no window.p5 dependency)
- ✅ Added: Proper peer dependencies declaration
- ✅ Enhanced: Full TypeScript type safety
- ✅ Improved: Documentation and installation guides

## License

ISC

