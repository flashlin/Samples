using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length != 2)
        {
            Console.WriteLine("Usage: ModifyFileTime <file_path> <datetime>");
            Console.WriteLine("Example: ModifyFileTime C:test.txt 2024-11-12T22:00:11");
            return;
        }

        string filePath = args[0];
        string dateTimeStr = args[1];

        if (!File.Exists(filePath))
        {
            Console.WriteLine($"File not found: {filePath}");
            return;
        }

        if (!DateTime.TryParse(dateTimeStr, out DateTime newDateTime))
        {
            Console.WriteLine($"Invalid datetime format: {dateTimeStr}");
            return;
        }

        try
        {
            SetFileTimes(filePath, newDateTime);
            Console.WriteLine($"File times updated to {newDateTime:yyyy-MM-dd HH:mm:ss}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    // Set file creation and last write time
    static void SetFileTimes(string filePath, DateTime dateTime)
    {
        // Set creation time
        File.SetCreationTime(filePath, dateTime);
        // Set last write time
        File.SetLastWriteTime(filePath, dateTime);
    }
}
