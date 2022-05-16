using Microsoft.AspNetCore.Razor.TagHelpers;
using System.Text;
using System.Text.Json;

namespace PizzaWeb.Models.Helpers
{
	[HtmlTargetElement("vue")]
	public class VueTagHelper : TagHelper
	{
		private readonly IWebHostEnvironment hostEnv;

		public VueTagHelper(IWebHostEnvironment hostEnv)
		{
			this.hostEnv = hostEnv;
		}
		
      public string Name { get; set; }
		
      public override void Process(TagHelperContext context, TagHelperOutput output)
      {
         output.TagName = "div";
         output.TagMode = TagMode.StartTagAndEndTag;

			var manifest = File.ReadAllText($"{hostEnv.WebRootPath}/dist/manifest.json");
			var dict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(manifest);
			var item = dict[$"spa/{Name}.html"];
			var jsFile = item.GetProperty("file").GetString();

			var sb = new StringBuilder();
         sb.AppendFormat($"<script src='/dist/{jsFile}'></script>", this.Name);

         output.PreContent.SetHtmlContent(sb.ToString());
      }
   }
}
