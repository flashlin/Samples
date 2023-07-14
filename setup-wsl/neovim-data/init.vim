call plug#begin('~/.config/nvim/plugged')
"底部狀態化
Plug 'bling/vim-airline'
"檔案瀏覽
Plug 'preservim/nerdtree'
Plug 'jistr/vim-nerdtree-tabs'
"快速跳轉
Plug 'easymotion/vim-easymotion'
"快速搜尋
Plug 'junegunn/fzf', { 'dir': '~/.fzf', 'do': './install --all' } 
Plug 'junegunn/fzf.vim'

" TypeScript 語法和重構插件
Plug 'neoclide/coc.nvim', {'branch': 'release'}

" C# LSP 插件
" Plug 'OmniSharp/omnisharp-vim'
call plug#end()

nnoremap <silent><leader>1 :source ~/AppData/Local/nvim/init.vim \| :PlugInstall<CR>

" 同時顯示行號和相對行號
let mapleader="\\"
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
" 設定游標所在行的顯示樣式
set cursorline 
highlight CursorLine cterm=NONE ctermfg=white ctermbg=darkgray guibg=darkgray guifg=white

" F3 開啟關閉
nnoremap <F3> :NERDTreeToggle<CR>
autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") && b:NERDTree.isTabTree()) | q | endif

" easymotion
nmap <space> <Plug>(easymotion-bd-w)

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


"
" fzf settings
" This is the default extra key bindings
let g:fzf_action = {
            \ 'ctrl-t': 'tab split',
            \ 'ctrl-x': 'split',
            \ 'ctrl-v': 'vsplit' }

" Default fzf layout
" - down / up / left / right
let g:fzf_layout = { 'down': '~70%' }

" Customize fzf colors to match your color scheme
let g:fzf_colors =
            \ { 'fg':      ['fg', 'Normal'],
            \ 'bg':      ['bg', 'Normal'],
            \ 'hl':      ['fg', 'Comment'],
            \ 'fg+':     ['fg', 'CursorLine', 'CursorColumn', 'Normal'],
            \ 'bg+':     ['bg', 'CursorLine', 'CursorColumn'],
            \ 'hl+':     ['fg', 'Statement'],
            \ 'info':    ['fg', 'PreProc'],
            \ 'prompt':  ['fg', 'Conditional'],
            \ 'pointer': ['fg', 'Exception'],
            \ 'marker':  ['fg', 'Keyword'],
            \ 'spinner': ['fg', 'Label'],
            \ 'header':  ['fg', 'Comment'] }

" Enable per-command history.
" CTRL-N and CTRL-P will be automatically bound to next-history and
" previous-history instead of down and up. If you don't like the change,
" explicitly bind the keys to down and up in your $FZF_DEFAULT_OPTS.
let g:fzf_history_dir = '~/.local/share/fzf-history'



" fzf 搜尋
nnoremap <leader>fl :Lines 
nnoremap <leader>fb :BLines 
nnoremap <leader>ff :Files 
nnoremap <leader>fg :GFiles 
nnoremap <leader>f? :GFiles? 
nnoremap <leader>ft :Tags<cr>
nnoremap <leader>fa :Ag 
nnoremap <leader>fc :Commits




" 啟用 LSP 支援 Typescript
autocmd FileType typescript,javascript,javascriptreact,typescriptreact CocStart
" 啟用 LSP 支援 C#
"autocmd FileType cs,sln nmap <buffer> <leader>d <Plug>(coc-definition)
"autocmd FileType cs,sln nmap <buffer> <leader>r <Plug>(coc-references)
"autocmd FileType cs,sln nmap <buffer> <leader>i <Plug>(coc-implementation)



