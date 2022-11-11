//https://github.com/requestly/modify-headers-manifest-v3/blob/master/src/background.ts

import rules from "./rules";

chrome.declarativeNetRequest.updateDynamicRules({
  removeRuleIds: rules.map((rule) => rule.id), // remove existing rules
  addRules: rules,
});
