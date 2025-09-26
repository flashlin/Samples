namespace CodeBoyLib.Services;

public interface IGenDatabaseModelWorkflow
{
    /// <summary>
    /// Builds database models for multiple target frameworks and generates a NuGet package
    /// </summary>
    /// <param name="buildParams">Build parameters</param>
    /// <returns>Generation and build result</returns>
    Task<GenDatabaseModelResult> Build(GenDatabaseModelBuildParams buildParams);
}