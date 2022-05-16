using Microsoft.CodeAnalysis;

namespace PizzaWeb.Models.Runtime;

public class CompileException : Exception
{
    public CompileException(IEnumerable<Diagnostic> failures)
    {
        Failures = failures;
    }

    public IEnumerable<Diagnostic> Failures { get; }
}