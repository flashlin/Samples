using System;
using Microsoft.CodeAnalysis;

namespace T1.GrpcProtoGenerator.Generators
{
    /// <summary>
    /// Simple logger interface for Source Generators
    /// </summary>
    internal interface ISourceGeneratorLogger
    {
        void LogDebug(string message);
        void LogInfo(string message);
        void LogWarning(string message);
        void LogError(string message);
        void LogError(string message, Exception exception);
    }

    /// <summary>
    /// Logger implementation for Source Generators that uses Roslyn's diagnostic reporting
    /// </summary>
    internal class SourceGeneratorLogger : ISourceGeneratorLogger
    {
        private readonly Action<Diagnostic> _reportDiagnostic;
        private readonly string _categoryName;

        public SourceGeneratorLogger(Action<Diagnostic> reportDiagnostic, string categoryName = "SourceGenerator")
        {
            _reportDiagnostic = reportDiagnostic ?? throw new ArgumentNullException(nameof(reportDiagnostic));
            _categoryName = categoryName;
        }

        public void LogDebug(string message)
        {
            var diagnostic = CreateDiagnostic(DiagnosticSeverity.Info, message);
            _reportDiagnostic(diagnostic);
        }

        public void LogInfo(string message)
        {
            var diagnostic = CreateDiagnostic(DiagnosticSeverity.Info, message);
            _reportDiagnostic(diagnostic);
        }

        public void LogWarning(string message)
        {
            var diagnostic = CreateDiagnostic(DiagnosticSeverity.Warning, message);
            _reportDiagnostic(diagnostic);
        }

        public void LogError(string message)
        {
            var diagnostic = CreateDiagnostic(DiagnosticSeverity.Error, message);
            _reportDiagnostic(diagnostic);
        }

        public void LogError(string message, Exception exception)
        {
            var fullMessage = $"{message} Exception: {exception}";
            var diagnostic = CreateDiagnostic(DiagnosticSeverity.Error, fullMessage);
            _reportDiagnostic(diagnostic);
        }

        private Diagnostic CreateDiagnostic(DiagnosticSeverity severity, string message)
        {
            var id = severity switch
            {
                DiagnosticSeverity.Error => "SG001",
                DiagnosticSeverity.Warning => "SG002", 
                DiagnosticSeverity.Info => "SG003",
                DiagnosticSeverity.Hidden => "SG004",
                _ => "SG000"
            };

            var title = $"Source Generator {severity}";
            var fullMessage = $"[{_categoryName}] {message}";

            var descriptor = new DiagnosticDescriptor(
                id: id,
                title: title,
                messageFormat: "{0}",
                category: "SourceGenerator",
                defaultSeverity: severity,
                isEnabledByDefault: true,
                description: "Source Generator diagnostic message"
            );

            return Diagnostic.Create(descriptor, Location.None, fullMessage);
        }
    }
}
