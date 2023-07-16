":PlugUpdate 安裝或更新
":PlugClean 清除沒有再用的套件
":PlugUpgrade 升級vim-plug 管理套件本身
":PlugStatus 查看狀態
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
"彩虹括號
Plug 'frazrepo/vim-rainbow'
"移動程式碼
Plug 'matze/vim-move'
"自動 :e
source ~/AppData/Local/nvim/wilder-setup.nvim

" TypeScript 語法和重構插件
Plug 'neoclide/coc.nvim', {'branch': 'release'}

" C# LSP 插件
" Plug 'OmniSharp/omnisharp-vim'
call plug#end()

" 1 重新載入
nnoremap <silent><leader>1 :source ~/AppData/Local/nvim/init.vim \| :PlugInstall<CR>

" 同時顯示行號和相對行號
let mapleader="\\"
set number
set relativenumber
set clipboard+=unnamedplus
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

"保存檔案
nnoremap <C-S> :w<CR>
inoremap <C-S> <Esc>:w<CR>i

source ~/AppData/Local/nvim/easymotion.vim
source ~/AppData/Local/nvim/NERDTreeToggle.vim
source ~/AppData/Local/nvim/rainbow.vim
source ~/AppData/Local/nvim/vim-move.vim
source ~/AppData/Local/nvim/wilder.vim
source ~/AppData/Local/nvim/fzf.vim


" 啟用 LSP 支援 Typescript
autocmd FileType typescript,javascript,javascriptreact,typescriptreact CocStart
" 啟用 LSP 支援 C#
"autocmd FileType cs,sln nmap <buffer> <leader>d <Plug>(coc-definition)
"autocmd FileType cs,sln nmap <buffer> <leader>r <Plug>(coc-references)
"autocmd FileType cs,sln nmap <buffer> <leader>i <Plug>(coc-implementation)



