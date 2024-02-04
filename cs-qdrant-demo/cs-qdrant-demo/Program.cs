// See https://aka.ms/new-console-template for more information

using Qdrant.Client;
using Qdrant.Client.Grpc;

Console.WriteLine("Hello, World!");

//var client = new QdrantClient("localhost", 6333);

var channel = QdrantChannel.ForAddress("https://localhost:6334", new ClientConfiguration
{
    ApiKey = "<api key>",
    CertificateThumbprint = "<certificate thumbprint>"
});
var grpcClient = new QdrantGrpcClient(channel);
var client = new QdrantClient(grpcClient);


await client.CreateCollectionAsync("qa_collection",
    new VectorParams
    {
        Size = 500,
        Distance = Distance.Cosine
    });

var random = new Random();
var dataList = new[]
{
    new PointStruct
    {
        Id = (ulong)1,
        Vectors = Enumerable.Range(1, 100).Select(_ => (float)random.NextDouble()).ToArray(),
        Payload =
        {
            ["short_id"] = "red",
        }
    }
};

var updateResult = await client.UpsertAsync("qa_collection", dataList);

var queryVector = Enumerable.Range(1, 100).Select(_ => (float)random.NextDouble()).ToArray();
// return the 5 closest points
var points = await client.SearchAsync(
    "qa_collection",
    queryVector,
    limit: 5);
    
Console.WriteLine($"points= {points}");