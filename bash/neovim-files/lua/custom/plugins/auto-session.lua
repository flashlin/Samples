-- 自動會話管理非常適合在退出 Neovim 之前自動儲存會話並在您回來後繼續工作。
return {
    "rmagatti/auto-session",
    config = function()
      local auto_session = require("auto-session")
  
      auto_session.setup({
        auto_restore_enabled = false,
        auto_session_suppress_dirs = { "~/", "~/Dev/", "~/Downloads", "~/Documents", "~/Desktop/" },
      })
  
      local keymap = vim.keymap
  
      keymap.set("n", "<leader>wr", "<cmd>SessionRestore<CR>", { desc = "Restore session for cwd" }) -- restore last workspace session for current directory
      keymap.set("n", "<leader>ws", "<cmd>SessionSave<CR>", { desc = "Save session for auto session root dir" }) -- save workspace session for current working directory
    end,
  }