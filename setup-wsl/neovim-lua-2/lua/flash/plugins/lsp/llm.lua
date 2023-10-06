return {
    "gsuuon/llm.nvim",
    lazy = true,
    config = function()
        prompts = {
            ['codellama'] = {
                provider = codellama,
                options = {
                    url = 'http://localhost:11434/api/generate'   
                }
            }
        }
    end
}