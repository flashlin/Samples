local overrides = require("custom.configs.overrides")

overrides.telescope = {
  defaults = {
    file_ignore_patterns = {
      "^.git/",
      "^node_modules/",
      "^bin/",
      "^obj/",
    },
  },
}

return overrides