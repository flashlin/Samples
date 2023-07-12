call plug#begin('~/.config/nvim/plugged')
"底部狀態化
Plug 'bling/vim-airline'
"檔案瀏覽
Plug 'preservim/nerdtree'
Plug 'jistr/vim-nerdtree-tabs'

" TypeScript 語法和重構插件
Plug 'neoclide/coc.nvim', {'branch': 'release'}
call plug#end()

" 同時顯示行號和相對行號
set number
set relativenumber
set encoding=utf-8
" 自动判断编码时,依次尝试下编码
set fileencodings=utf-8,ucs-bom,big5
set autoindent
set smartindent
set tabstop=3
set shiftwidth=3
" 設定 space 取代 tab
set expandtab

" F3 開啟關閉
nnoremap <F3> :NERDTreeToggle<CR>
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

" 搜索的时候不區分大小写,是set ignorecase 縮寫.如果你想啟用,输入:set noic(noignorecase缩写)
set ic
" 搜索的时候随字符高亮
set hlsearch

"ctrl+a	全選+複製
map <C-A> ggVG
map! <C-A> <Esc>ggVG
"ctrl+y 複製到系統剪貼簿
map  <C-Y> "+y
map! <C-Y> "+y


" 啟用 LSP 支援
autocmd FileType typescript,javascript,javascriptreact,typescriptreact CocStart