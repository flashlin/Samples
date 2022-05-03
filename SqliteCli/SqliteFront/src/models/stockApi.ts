export class StockApi {
  getTransListAsync() {
    return this.getAsync("Stock/GetTransList");
  }

  private getAsync(endpoint: string) {
    return fetch(`http://sqlite.localdev.net:3000/${endpoint}`, {
      method: "GET",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        //"Authorization": `Bearer ${token}`,
      },
    }).then((response) => response.json());
  }

  postAsync(endpoint: string, data: object) {
    return fetch(`http://127.0.0.1:3000/${endpoint}`, {
      method: "POST",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        //"Authorization": `Bearer ${token}`,
      },
      body: JSON.stringify(data),
    }).then((response) => {
      return response.json();
    });
  }
}
