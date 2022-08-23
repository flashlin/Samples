using System.Text.RegularExpressions;
using T1.Standard.Extensions;
using T1.Standard.Serialization;

namespace WCodeSnippetX.Models;

public class CodeSnippetService : ICodeSnippetService
{
    private readonly ICodeSnippetRepo _repo;
    private readonly IJsonSerializer _jsonSerializer;

    public CodeSnippetService(ICodeSnippetRepo repo, IJsonSerializer jsonSerializer)
    {
        _jsonSerializer = jsonSerializer;
        _repo = repo;
    }

    public void DeleteCodeById(int id)
    {
       _repo.DeleteCodeById(id);
    }

	 public void UpdateCode(CodeSnippetEntity code)
    {
       _repo.UpdateCode(code);
    }

    public void AddCode(CodeSnippetEntity code)
    {
       _repo.AddCode(code);
    }

    public List<CodeSnippetEntity> Query(string text)
    {
        var codeSnippets = _repo.QueryCode(text)
            .ToList();

        if (string.IsNullOrEmpty(text))
        {
            return codeSnippets;
        }

        var patterns = text.ParseCommandArgsLine().ToList();
        var prevCodeSnippets = codeSnippets;
        var filterSnippets = new List<CodeSnippetEntity>();
        foreach (var pattern in patterns)
        {
            filterSnippets = new List<CodeSnippetEntity>();
            foreach (var codeSnippet in prevCodeSnippets)
            {
                if (IsMatch(codeSnippet, pattern))
                {
                    filterSnippets.Add(codeSnippet);
                }
            }
            prevCodeSnippets = filterSnippets;
        }
        return filterSnippets;
    }


    private bool IsMatch(CodeSnippetEntity codeSnippet, string pattern)
    {
        if (CodeSnippetRepo.DefaultProgramLanguages.Contains(pattern) && 
            Regex.Match(codeSnippet.ProgramLanguage, pattern).Success)
        {
            return true;
        }

        if (Regex.Match(codeSnippet.Content, pattern).Success)
        {
            return true;
        }

        if (Regex.Match(codeSnippet.Description, pattern).Success)
        {
            return true;
        }

        return false;
    }
}