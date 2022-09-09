using System.Buffers;
using System.Net;
using System.Net.Sockets;
using T1.ForwardOk.Sockets;

namespace T1.ForwardOk;

public class AsyncSocket : IAsyncSocket
{
    private readonly Socket _socket;
    private readonly byte[] _readBuffer;
    private readonly int _bufferSize;

    public AsyncSocket(Socket socket, int bufferSize)
    {
        _bufferSize = bufferSize;
        _socket = socket;
        _readBuffer = ArrayPool<byte>.Shared.Rent(bufferSize);
    }

    public void BeginReceive(ReceiveAsyncCallback receiveCallback, Action closeCallback)
    {
        var state = new AsyncSocketState
        {
            Buffer = _readBuffer,
            ReceiveAsyncCallback = receiveCallback,
            ClosedCallback = closeCallback,
        };
        _socket.BeginReceive(state.Buffer, 0, state.Buffer.Length, 0, OnDataReceive, state);
    }

    public void Close()
    {
        _socket.Close();
        ArrayPool<byte>.Shared.Return(_readBuffer);
    }

    public void SendAsync(byte[] buffer, int bytes)
    {
        _socket.Send(buffer, bytes, SocketFlags.None);
    }

    public void SendMessage(string message)
    {
        var buff = message.FastToBytes();
        _socket.Send(buff, buff.Length, SocketFlags.None);
    }

    private static void OnDataReceive(IAsyncResult result)
    {
        var state = (AsyncSocketState) result.AsyncState!;
        try
        {
            var bytesRead = state.Socket.EndReceive(result);
            if (bytesRead <= 0) return;
            state.ReceiveAsyncCallback(state.Buffer, bytesRead);
            state.Socket.BeginReceive(state.Buffer, 0, state.Buffer.Length, 0, OnDataReceive, state);
        }
        catch
        {
            state.ClosedCallback();
            state.Socket.Close();
        }
    }

    public Task ConnectAsync(IPEndPoint endpoint)
    {
        throw new NotImplementedException();
    }

    public void SetupReceive(ReceiveAsyncCallback receiveAsyncCallback, ClosedAsyncCallback closedAsyncCallback)
    {
        throw new NotImplementedException();
    }

    public Task CloseAsync()
    {
        throw new NotImplementedException();
    }

    public Task SendAsync(byte[] buffer, int offset, int bytes)
    {
        throw new NotImplementedException();
    }

    public Task SendMessageAsync(string message)
    {
        throw new NotImplementedException();
    }
}