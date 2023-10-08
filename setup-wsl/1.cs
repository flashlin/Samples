using System;

class Program {
    static void Main() {
        Console.WriteLine("Hello World!");
        Console.WriteLine("Result:");
    }
}

public class TypeClass {
    public string name { get; set; }
    public int id { get; set; }
}


public static async Task<string> SendPostJsonRequestAsync(string url, string jsonData)
{
  var client = new HttpClient()

  client.DefaultRequestHeaders.Accept.Clear();
  client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

  var content = new StringContent(jsonData, Encoding.UTF8, "application/json");
  await client.SendAsync(url, content);

  var response = await client.GetResponseAsync();
  using (var streamReader = new StreamReader(await response.Content.ReadAsStreamAsync(), Encoding.UTF8))
  {
    return await streamReader.ReadToEndAsync();
  }

}

