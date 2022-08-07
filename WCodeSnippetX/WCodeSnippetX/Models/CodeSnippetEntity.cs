using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace WCodeSnippetX.Models;
//Microsoft.EntityFrameworkCore
[Table("CodeSnippets")]
public class CodeSnippetEntity
{
	[Key]
	public int Id { get; set; }
	[StringLength(20)]
	public string ProgramLanguage { get; set; } = string.Empty;
	[StringLength(4000)]
	public string Content { get; set; } = string.Empty;
	[StringLength(1000)]
	public string Description { get; set; } = string.Empty;
}

public class CodeSnippetService
{
	public void Query(string text)
	{

	}
}