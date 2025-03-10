﻿using System.Net;
using Microsoft.Extensions.DependencyInjection;
using T1.ForwardOk;

var services = new ServiceCollection();
services.AddSingleton<TcpForwarderSlim>();

var serviceProvider = services.BuildServiceProvider();
Console.WriteLine("Hello, World!");

new GrpcClientConnection();
//var remoteWebForward = serviceProvider.GetService<IRemoteForwardConnection>();
var tcpForwarderSlim = serviceProvider.GetService<TcpForwarderSlim>()!;

await tcpForwarderSlim.StartAsync("localhost:3000", "127.0.0.1:3001");

Console.WriteLine("END");
Console.ReadLine();