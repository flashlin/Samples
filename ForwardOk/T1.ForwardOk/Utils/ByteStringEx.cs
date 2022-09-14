using Google.Protobuf;

namespace T1.ForwardOk.Utils;

public static class ByteStringEx
{
    public static ByteString ToByteString(this Stream stream)
    {
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        var bytes = memoryStream.ToArray();
        return ByteString.CopyFrom(bytes);
    }
    
    public static ByteString ToByteString(this byte[] bytes, int offset, int length)
    {
        var newBytes = bytes.AsSpan().Slice(offset, length);
        return ByteString.CopyFrom(newBytes);
    }
}