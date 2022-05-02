using System;
using System.Runtime.InteropServices;

namespace T1.SqlLocalData;

public class LocalDbFactory
{
    public ISqlLocalDb Create(string dataFolder)
    {
        if (GetOSPlatform() == OSPlatform.Linux)
        {
            return new LinuxLocalDb(null);
        }
        return new SqlLocalDb(dataFolder);
    }

    private OSPlatform GetOSPlatform()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return OSPlatform.Linux;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return OSPlatform.OSX;
        }

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return OSPlatform.Windows;
        }

        throw new Exception("Cannot determine operating system!");
    }
}