using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace CheckAssemblyList
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            var path = args[0];
            var files = Directory.GetFiles(path, "*.dll");
            var assemblyHelper = new AssemblyHelper();
            foreach (var file in files)
            {
                Console.WriteLine($"{file}");
                assemblyHelper.InspectDependencies(file);
            }
        }
    }

    class AssemblyHelper
    {
        public void InspectDependencies(string dllPath)
        {
            var targetAssembly = Assembly.LoadFrom(dllPath);
            Console.WriteLine($"{targetAssembly.Location} AssemblyInfo='{GetAssemblyInfo(targetAssembly)}'");
            var dependencies = targetAssembly.GetReferencedAssemblies();
            foreach (var dependency in dependencies)
            {

                var assemblyInfo = "";
                try
                {
                    var assembly = Assembly.Load(dependency);
                    assemblyInfo = GetAssemblyInfo(assembly);
                }
                catch
                {
                    assemblyInfo = "";
                }
                
                Console.WriteLine($"    {dependency.Name}, Version={dependency.Version} AssemblyInfo='{assemblyInfo}'");
            }
        }

        public string GetAssemblyInfo(Assembly executingAssembly)
        {
            var versionAttribute = executingAssembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>();
            if (versionAttribute != null)
            {
                return versionAttribute.InformationalVersion;
            }
            return "";
        }
        
        public string GetFileVersion(AssemblyName dependency)
        {
            try
            {
                var fileVersionInfo = FileVersionInfo.GetVersionInfo(dependency.Name + ".dll");
                return fileVersionInfo.FileVersion;
            }
            catch (FileNotFoundException)
            {
                return "";
            }
        }
    }
}