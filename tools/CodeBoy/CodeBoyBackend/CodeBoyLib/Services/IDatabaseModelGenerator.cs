namespace CodeBoyLib.Services;

public interface IDatabaseModelGenerator
{
    /// <summary>
    /// Generates EF Code First models from database using scaffolding
    /// </summary>
    /// <param name="parameters">Database generation parameters</param>
    /// <returns>EF generation output containing csproj path and code files</returns>
    Task<EfGenerationOutput> GenerateEfCode(DatabaseGenerationParams parameters);
}