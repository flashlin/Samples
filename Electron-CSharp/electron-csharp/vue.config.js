module.exports = {
  pluginOptions: {
    electronBuilder: {
      builderOptions: {
        files: [
          {
            filter: ["**/*"]
          }
        ],
        extraFiles: ["./extensions/"],
        asar: false
      },
      mainProcessFile: "src/background.ts",
      mainProcessWatch: ["src/main"],
      // [1.0.0-rc.4+] Provide a list of arguments that Electron will be launched with during "electron:serve",
      // which can be accessed from the main process (src/background.js).
      // Note that it is ignored when --debug flag is used with "electron:serve", as you must launch Electron yourself
      mainProcessArgs: [],
      productionSourceMap: false
    }
  }
};
