using WCodeSnippetX.Models.Repos;

namespace WCodeSnippetX.Models;

public class CodeSnippetRepo
{
	private readonly CodeSnippetDbContext _dbContext;

	public CodeSnippetRepo(CodeSnippetDbContext dbContext)
	{
		_dbContext = dbContext;
	}

	public void Query(string text)
	{

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

public class CodeSnippetService
{
	public void Query(string text)
	{

	}
}