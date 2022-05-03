using SqliteCli.Factories;

namespace SqliteCli;

public static class EnvironmentExtension
{
    public static string GetEnvironmentVariableOrDefault(this IEnvironment environment, string name,
        string defaultValue)
    {
        var variable = environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(variable))
        {
            variable = defaultValue;
        }

        return variable;
    }
}