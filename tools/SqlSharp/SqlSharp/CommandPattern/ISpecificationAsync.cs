namespace SqlSharp.CommandPattern;

public interface ISpecificationAsync<in TArgs, TReturn>
{
    bool IsMatch(TArgs args);
    Task<TReturn> ExecuteAsync(TArgs args);
}

public class SpecificationAsyncEvaluator<TArgs,TReturn>
{
    private readonly List<ISpecificationAsync<TArgs,TReturn>> _rules;

    public SpecificationAsyncEvaluator(IEnumerable<ISpecificationAsync<TArgs, TReturn>> rules)
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