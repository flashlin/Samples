namespace CodeBoyLib.Services;

public interface IGenDatabaseModelWorkflow
{
    /// <summary>
    /// Builds database models for multiple target frameworks and generates a NuGet package
    /// </summary>
    /// <param name="buildParams">Build parameters</param>
    /// <returns>Generation and build result</returns>
    Task<GenDatabaseModelResult> Build(GenDatabaseModelBuildParams buildParams);

    /// <summary>
    /// Generates EF Code First models for a single target framework
    /// </summary>
    /// <param name="buildParams">Build parameters</param>
    /// <param name="framework">Target framework version</param>
    /// <returns>EF generation output</returns>
    Task<EfGenerationOutput> GenEfCodeFirst(GenDatabaseModelBuildParams buildParams, string framework);
}