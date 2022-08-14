using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore;
using T1.Standard.Extensions;
using WCodeSnippetX.Models.Repos;

namespace WCodeSnippetX.Models;

public class CodeSnippetRepo : ICodeSnippetRepo
{
	private readonly CodeSnippetDbContext _dbContext;

	private static readonly string[] DefaultProgramLanguages = new[]
	{
		"cs",
		"ts",
		"js",
		"py",
		"go",
		"csharp",
		"typescript",
		"javascript",
		"python",
		"java",
		"xml",
		"json",
		"c++",
		"c",
		"c#",
		"cpp",
		"txt",
		"md",
		"markdown",
	};

	public CodeSnippetRepo(CodeSnippetDbContext dbContext)
	{
		_dbContext = dbContext;
	}

	public IEnumerable<CodeSnippetEntity> QueryCode(string text)
	{
		var codeSnippets = _dbContext.CodeSnippets.ToList();
		if (string.IsNullOrEmpty(text))
		{
			foreach (var codeSnippet in codeSnippets)
			{
				yield return codeSnippet;
			}
		}

		var patterns = text.ParseCommandArgsLine().ToList();
		foreach (var codeSnippet in codeSnippets)
		{
			if (IsMatch(codeSnippet, patterns))
			{
				yield return codeSnippet;
			}
		}
	}

	public void UpdateCode(CodeSnippetEntity codeSnippet)
	{
		_dbContext.CodeSnippets.Update(codeSnippet);
		_dbContext.SaveChanges();
	}

	public void AddCode(CodeSnippetEntity codeSnippet)
	{
		_dbContext.CodeSnippets.Add(codeSnippet);
		_dbContext.SaveChanges();
	}

	public void DeleteCode(CodeSnippetEntity codeSnippet)
	{
		_dbContext.CodeSnippets.Remove(codeSnippet);
		_dbContext.SaveChanges();
	}

	public void DeleteCodeById(int id)
	{
		var item = new CodeSnippetEntity()
		{
			Id = id,
		};
		var entry = _dbContext.Entry(item);
		entry.State = EntityState.Deleted;
		_dbContext.SaveChanges();
	}

	private bool IsMatch(CodeSnippetEntity codeSnippet, List<string> patterns)
	{
		foreach (var pattern in patterns)
		{
			if (DefaultProgramLanguages.Contains(pattern) && !Regex.Match(codeSnippet.ProgramLanguage, pattern).Success)
			{
				return false;
			}

			if (Regex.Match(codeSnippet.Content, pattern).Success)
			{
				return true;
			}

			if (Regex.Match(codeSnippet.Description, pattern).Success)
			{
				return true;
			}
		}
		return false;
	}

	private void Initialize()
	{
		var baseDir = AppDomain.CurrentDomain.BaseDirectory;
		var dbFile = Path.Combine(baseDir, CodeSnippetDbContext.DbFilename);
		if (File.Exists(dbFile))
		{
			return;
		}
	}
}