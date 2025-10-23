using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace T1.EfCodeFirstGenerateCli.Common
{
    public static class FileWriterHelper
    {
        /// <summary>
        /// Write generated files to disk, skipping existing files
        /// </summary>
        /// <param name="generatedFiles">Dictionary of relative path -> file content</param>
        /// <param name="generatedDir">Base directory for generated files</param>
        /// <param name="logAction">Optional logging action</param>
        /// <returns>Number of files written</returns>
        public static int WriteGeneratedFiles(
            Dictionary<string, string> generatedFiles,
            string generatedDir,
            Action<string>? logAction = null)
        {
            int writtenCount = 0;
            int skippedCount = 0;
            
            foreach (var kvp in generatedFiles)
            {
                var filePath = Path.Combine(generatedDir, kvp.Key);
                var fileDir = Path.GetDirectoryName(filePath);
                
                if (!string.IsNullOrEmpty(fileDir))
                {
                    Directory.CreateDirectory(fileDir);
                }
                
                // Skip if file already exists
                if (File.Exists(filePath))
                {
                    logAction?.Invoke($"  Skipped (already exists): {kvp.Key}");
                    skippedCount++;
                    continue;
                }
                
                File.WriteAllText(filePath, kvp.Value, Encoding.UTF8);
                writtenCount++;
            }
            
            if (skippedCount > 0)
            {
                logAction?.Invoke($"  Skipped {skippedCount} existing file(s).");
            }
            
            return writtenCount;
        }
    }
}

