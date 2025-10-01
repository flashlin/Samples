# VimLibrary

A TypeScript Web Component Library for building Vim-like editor interfaces.

## Features

- 🎯 Built with TypeScript for type safety
- 📦 Bundled with Vite for optimal performance
- 🎨 Web Components for framework-agnostic usage
- 🔧 Easy to integrate into any frontend project

## Installation

```bash
# From VimFront project
pnpm add vim-library

# Or use local path during development
pnpm add file:../VimLibrary
```

## Usage

### ES Module

```typescript
import { VimEditor } from 'vim-library'

// Create and use the component
const editor = new VimEditor()
editor.init({
  initialContent: 'Hello, Vim!'
})
document.body.appendChild(editor)
```

### HTML (UMD)

```html
<script src="./vim-library.umd.js"></script>
<vim-editor></vim-editor>

<script>
  const editor = document.querySelector('vim-editor')
  editor.init({ initialContent: 'Hello, Vim!' })
</script>
```

## Development

### Setup

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Type checking
pnpm type-check
```

### Build

```bash
# Build the library
pnpm build
```

This will generate:
- `dist/vim-library.es.js` - ES module build
- `dist/vim-library.umd.js` - UMD build for browsers
- `dist/index.d.ts` - TypeScript type definitions

## API

### VimEditor

The main editor component.

#### Methods

- `init(options?: VimEditorOptions)` - Initialize the editor
- `getContent(): string` - Get current content
- `setContent(content: string)` - Set content

#### Types

```typescript
interface VimEditorOptions {
  initialContent?: string
  config?: VimConfig
}

interface VimConfig {
  mode?: 'normal' | 'insert' | 'visual'
  readOnly?: boolean
  lineNumbers?: boolean
}
```

## License

ISC

