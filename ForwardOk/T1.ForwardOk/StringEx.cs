using System.Runtime.InteropServices;

namespace T1.ForwardOk;

public static class StringEx
{
    public static byte[] FastToBytes(this string str)
    {
        var charSpan = str.AsMemory();
        if (charSpan.Length == 0)
        {
            return Array.Empty<byte>();
        }
        var array = new byte[charSpan.Length];
        unsafe
        {
            var ptr = Marshal.StringToHGlobalAnsi(str);
            Marshal.Copy(ptr, array, 0, charSpan.Length);
            return array;
        }
    }
}