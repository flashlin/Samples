local wk = require "which-key"

wk.register({
  C = {
    c = { "<cmd>ChatGPT<CR>", "ChatGPT" },
    e = { "<cmd>ChatGPTEditWithInstruction<CR>", "Edit with instruction", mode = { "n", "v" } },
    g = { "<cmd>ChatGPTRun grammar_correction<CR>", "Grammar correction", mode = { "n", "v" } },
    t = { "<cmd>ChatGPTRun translate<CR>", "Translate", mode = { "n", "v" } },
    k = { "<cmd>ChatGPTRun keywords<CR>", "Keywords", mode = { "n", "v" } },
    d = { "<cmd>ChatGPTRun docstring<CR>", "Docstring", mode = { "n", "v" } },
    a = { "<cmd>ChatGPTRun add_tests<CR>", "Add tests", mode = { "n", "v" } },
    o = { "<cmd>ChatGPTRun optimize_code<CR>", "Optimize code", mode = { "n", "v" } },
    s = { "<cmd>ChatGPTRun summarize<CR>", "Summarize", mode = { "n", "v" } },
    f = { "<cmd>ChatGPTRun fix_bugs<CR>", "Fix bugs", mode = { "n", "v" } },
    x = { "<cmd>ChatGPTRun explain_code<CR>", "Explain code", mode = { "n", "v" } },
    w = { "<cmd>ChatGPTRun complete_code<CR>", "Complete code", mode = { "n", "v" } },
    l = { "<cmd>ChatGPTRun code_readability_analysis<CR>", "Code readability analysis", mode = { "n", "v" } },
    D = { "<cmd>Codeium Disable<CR>", "Disable Codeium", mode = { "n", "v" } },
    E = { "<cmd>Codeium Enable<CR>", "Enable Codeium", mode = { "n", "v" } },
  },
  s = {
    f = {
      function()
        require("flash").treesitter()
      end,
      "Select function",
      mode = { "n", "x", "o" },
    },
  },
}, { prefix = "<leader>" })