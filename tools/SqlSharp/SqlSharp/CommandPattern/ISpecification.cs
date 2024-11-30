namespace SqlSharp.CommandPattern;

public interface ISpecification<in TArgs, TReturn>
{
    bool IsMatch(TArgs args);
    Task<TReturn> ExecuteAsync(TArgs args);
}

public class SpecificationEvaluator<TArgs,TReturn>
{
    private readonly List<ISpecification<TArgs,TReturn>> _rules;

    public SpecificationEvaluator(IEnumerable<ISpecification<TArgs, TReturn>> rules)
    {
        _rules = rules.ToList();
    }

    public Task<TReturn> EvaluateAsync(TArgs args)
    {
        foreach (var rule in _rules)
        {
            if (rule.IsMatch(args))
            {
                return rule.ExecuteAsync(args);
            }
        }
        throw new KeyNotFoundException();
    }
}