using NUnit.Framework;
using System.Text.Json;
using System.Text.Json.Nodes;
using PizzaWeb.Models.Banner;

namespace TestProject
{
	public class JsonTest
	{
		[SetUp]
		public void Setup()
		{
		}

		[Test]
		public void ToDictionaryTest()
		{
			var jsonStr = @"{
  ""spa/luncher.html"": {
    ""file"": ""assets/luncher.ddee1e2b.js"",
    ""src"": ""spa/luncher.html"",
    ""isEntry"": true
  }
}";

			var dict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(jsonStr);
			var item = dict[$"spa/luncher.html"];
			var jsFile = item.GetProperty("file").GetString();
			
			Assert.That(jsFile, Is.EqualTo("assets/luncher.ddee1e2b.js"));
		}
		
		[Test]
		public void ToDictionaryCTest()
		{
			var jsonStr = @"
[
	{
		'name': 'Title',
		'tempVarType': 0
	},
	{
		'name': 'Price',
		'tempVarType': 1
	},
]";
			var variablesJson = jsonStr.Replace("'", "\"");

			var jsonOptions = new JsonSerializerOptions
			{
				AllowTrailingCommas = true,
				PropertyNamingPolicy = JsonNamingPolicy.CamelCase
			};
			var varitablesList = JsonSerializer.Deserialize<List<TemplateVariable>>(variablesJson, jsonOptions);

			var item0 = varitablesList[0];
			
			Assert.That(item0.Name, Is.EqualTo("Title"));
		}
		
	}

   public static class JsonDictStringObjExtensions
   {
      public static Dictionary<string, object> ToStringObjectDictionary(this JsonObject jsonObject)
      {
         var dict = new Dictionary<string, object>();
         foreach (var prop in jsonObject)
         {
            object value;
            if (prop.Value == null) value = null!;
            else if (prop.Value is JsonArray) value = prop.Value.AsArray();
            else if (prop.Value is JsonObject) value = prop.Value.AsObject();
            else
            {
               var v = prop.Value.AsValue();
               var t = prop.Value.ToJsonString();
               if (t.StartsWith('"'))
               {
                  if (v.TryGetValue<DateTime>(out var d)) value = d;
                  else if (v.TryGetValue<Guid>(out var g)) value = g;
                  else value = v.GetValue<string>();
               }
               else value = v.GetValue<decimal>();
            }
            dict.Add(prop.Key, value);
         }
         return dict;
      }
   }
}