using System.Text.RegularExpressions;
using Microsoft.EntityFrameworkCore;
using T1.Standard.Extensions;
using WCodeSnippetX.Models.Repos;

namespace WCodeSnippetX.Models;

public class CodeSnippetRepo : ICodeSnippetRepo
{
    private readonly CodeSnippetDbContext _dbContext;

    public static readonly string[] DefaultProgramLanguages = new[]
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
        if (string.IsNullOrEmpty(text))
        {
            var codeSnippets = _dbContext.CodeSnippets.ToList();
            foreach (var codeSnippet in codeSnippets)
            {
                yield return codeSnippet;
            }
            yield break;
        }

        var patterns = text.ParseCommandArgsLine().ToList();
        var queryLang = patterns
            .FirstOrDefault(x => DefaultProgramLanguages.Contains(x));
        if (queryLang == null)
        {
            var codeSnippets = _dbContext.CodeSnippets.ToList();
            foreach (var codeSnippet in codeSnippets)
            {
                yield return codeSnippet;
            }
            yield break;
        }

        var codeSnippetsByLang = _dbContext.CodeSnippets
            .Where(x => x.ProgramLanguage == queryLang);

        foreach (var entity in codeSnippetsByLang)
        {
            yield return entity;
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
        //not work ??
        //var item = new CodeSnippetEntity()
        //{
        //	Id = id,
        //};
        //_dbContext.Entry(item).State = EntityState.Deleted;
        //_dbContext.SaveChanges();

        var item = _dbContext.CodeSnippets.AsTracking()
            .First(x => x.Id == id);
        _dbContext.CodeSnippets.Remove(item);
        _dbContext.SaveChanges();
    }

    private bool IsMatch(CodeSnippetEntity codeSnippet, IEnumerable<string> patterns)
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