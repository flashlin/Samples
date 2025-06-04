using T1.SqlSharp.Expressions;

public interface IGenerateTrainDataService
{
    IEnumerable<ISqlExpression> EnumerateDataTableSchemas(string sqlFolder);
    void Run();
}