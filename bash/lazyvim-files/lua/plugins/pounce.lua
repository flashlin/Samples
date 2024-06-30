return {
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
}