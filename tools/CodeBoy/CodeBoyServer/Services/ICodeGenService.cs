using CodeBoyServer.Models;

namespace CodeBoyServer.Services
{
    /// <summary>
    /// Interface for code generation service
    /// </summary>
    public interface ICodeGenService
    {
        /// <summary>
        /// Generate Web API client code
        /// </summary>
        /// <param name="args">Generation arguments</param>
        /// <returns>Generated code</returns>
        Task<string> GenerateCode(GenWebApiClientArgs args);
    }
}
