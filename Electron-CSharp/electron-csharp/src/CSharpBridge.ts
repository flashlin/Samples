const { ConnectionBuilder } = require("electron-cgi");

let connection = new ConnectionBuilder()
  .connectTo("dotnet", "run", "--project", "../DotnetBridge/DotnetBridge")
  .build();

connection.onDisconnect = () => {
  console.log("lost");
};

connection.send("greeting", "Mom", (response: any) => {
  console.log(response);
  connection.close();
});