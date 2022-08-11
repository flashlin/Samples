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

	public string Query(string text)
	{
		return _jsonSerializer.Serialize(_repo.QueryCode(text));
	}
}
