using System.Net;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

public static class DnsEx
{
    public static ConfiguredTaskAwaitable<IPAddress[]> GetAddressesAsync(this string hostNameOrAddress)
    {
        return Dns.GetHostAddressesAsync(hostNameOrAddress).ConfigureAwait(false);
    }

    public static async IAsyncEnumerable<IPEndPoint> ParseAddressesAsync(this string serverNameOrAddress)
    {
        var rg = new Regex(@"(<server>:\d+\.\d+\.\d+\.\d+)(<port>:\:?\d+)", RegexOptions.Compiled);
        var match = rg.Match(serverNameOrAddress);
        if (!match.Success)
        {
            throw new InvalidFormatException($"Invalid '{serverNameOrAddress}'.");
        }

        var server = match.Groups["server"].Value;
        var port = int.Parse(match.Groups["port"].Value);
        var ipAddresses = await server.GetAddressesAsync();
        foreach (var ipAddress in ipAddresses)
        {
            var endpoint = new IPEndPoint(ipAddress, port);
            yield return endpoint;
        }
    }
}