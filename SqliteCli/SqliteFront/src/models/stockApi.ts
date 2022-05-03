export class StockApi {
  getApi() {
    return fetch("http://sqlite.localdev.net:3000/Stock/GetTransList", {
      method: "GET",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        //"Authorization": `Bearer ${token}`,
      },
    }).then(response => response.json());
  }
}