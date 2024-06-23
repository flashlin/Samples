return {
  {
    "stevearc/conform.nvim",
    -- event = 'BufWritePre', -- uncomment for format on save
    config = function()
      require "configs.conform"
    end,
  },
  {
    "kylechui/nvim-surround",
    version = "*", -- Use for stability; omit to use `main` branch for the latest features
    event = "VeryLazy",
    config = function()
      return require("custom.configs.nvim-surround").setup()
    end,
  },
  {
    "rlane/pounce.nvim",
    lazy = false,
    config = function()
      require'pounce'.setup{
        accept_keys = "JFKDLSAHGNUVRBYTMICEOXWPQZ",
        accept_best_key = "<enter>",
        multi_window = true,
        debug = false,
      }
    end,
  },
  {
    "mg979/vim-visual-multi",
    lazy = false,
  },
  {
    "folke/which-key.nvim",
    opts = require "custom.configs.whichkey",
  },

  -- These are some examples, uncomment them if you want to see them work!
  {
    "neovim/nvim-lspconfig",
    config = function()
      require("nvchad.configs.lspconfig").defaults()
      require "custom.configs.lspconfig"
    end,
  },
  -- {
  --   "williamboman/mason.nvim",
  --   opts = require "custom.configs.mason",
  -- },
  
  {
  	"williamboman/mason.nvim",
  	opts = {
  		ensure_installed = {
  			"lua-language-server", "stylua",
  			"html-lsp", "css-lsp" , "prettier",
        "typescript-language-server",
  		},
  	},
  },
  
  {
  	"nvim-treesitter/nvim-treesitter",
  	opts = {
  		ensure_installed = {
  			"vim", "lua", "vimdoc",
        "html", "css", "javascript"
  		},
  	},
  },
}
