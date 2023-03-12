namespace QueryWasm;

public class DriveHelper
{
    public static string[] GetDrives()
    {
        return DriveInfo.GetDrives()
            .Select(d => d.Name.Substring(0, 1))
            .GroupBy(x => x)
            .Select(x => x.Key)
            .ToArray();
    }
}