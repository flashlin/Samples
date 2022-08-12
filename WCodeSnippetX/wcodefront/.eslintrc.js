/** @format */

module.exports = {
  parser: 'vue-eslint-parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module',
    parser: '@typescript-eslint/parser',
  },
  extends: [
    'plugin:vue/vue3-essential',
    'eslint:recommended', 
    '@vue/eslint-config-typescript'],
  plugins: ['@typescript-eslint', 'vue', 'prettier'],
  env: {
    browser: true,
    node: true,
  },
  rules: {
    //'@typescript-eslint/rule-name': 'error',
    '@typescript-eslint/no-unused-vars': ['error'],
  },
};
