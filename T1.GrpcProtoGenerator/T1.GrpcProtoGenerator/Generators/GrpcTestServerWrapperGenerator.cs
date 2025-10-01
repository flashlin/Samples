using Microsoft.CodeAnalysis;
using T1.GrpcProtoGenerator.Generators.Models;

namespace T1.GrpcProtoGenerator.Generators
{
    [Generator]
    public class GrpcTestServerWrapperGenerator : IIncrementalGenerator
    {
        public void Initialize(IncrementalGeneratorInitializationContext context)
        {
            var protoFiles = context.AdditionalTextsProvider
                .Where(f => f.Path.EndsWith(".proto"));

            var protoFilesWithContent = protoFiles.Select((text, _) => new ProtoFileInfo
            {
                Path = text.Path,
                Content = text.GetText()!.ToString()
            });
            
            // Collect all proto files and process them together to handle imports
            var allProtoFiles = protoFilesWithContent.Collect();
            
            // Get compilation provider to check for package references
            var compilation = context.CompilationProvider;
            
            // Combine proto files with compilation information
            var protoFilesWithCompilation = allProtoFiles.Combine(compilation);

            context.RegisterSourceOutput(protoFilesWithCompilation, (spc, data) =>
            {
                var (allProtos, compilation) = data;
                var logger = InitializeLogger(spc);
                logger.LogWarning($"Starting source generation for {allProtos.Length} proto files");
                
                var combinedModel = new ProtoModelResolver().CreateCombinedModel(allProtos);

                logger.LogInfo("Source generation completed successfully");
            });
        }

        /// <summary>
        /// Initialize logger for source generation
        /// </summary>
        private ISourceGeneratorLogger InitializeLogger(SourceProductionContext spc)
        {
            return new SourceGeneratorLogger(spc.ReportDiagnostic, nameof(GrpcServerWrapperGenerator));
        }
    }
}

