using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

namespace T1.GrpcProtoGenerator.Generators;

/// <summary>
/// Resolver for creating combined proto models from multiple proto files
/// </summary>
public class ProtoModelResolver
{
    /// <summary>
    /// Create a combined model with unique messages and enums from all proto files
    /// </summary>
    public ProtoModel CreateCombinedModel(ImmutableArray<ProtoFileInfo> allProtos)
    {
        var combinedModel = new ProtoModel();
        var collectedElements = CollectAllProtoElements(allProtos);
            
        AddUniqueMessagesToModel(combinedModel, collectedElements.Messages);
        AddUniqueEnumsToModel(combinedModel, collectedElements.Enums);
        ProcessAndAddServices(combinedModel, collectedElements.Services);
            
        return combinedModel;
    }

    /// <summary>
    /// Collect all proto elements from all proto files
    /// </summary>
    private ProtoElementCollection CollectAllProtoElements(ImmutableArray<ProtoFileInfo> allProtos)
    {
        var collection = new ProtoElementCollection();
            
        foreach (var protoInfo in allProtos)
        {
            var model = ProtoParser.ParseProtoText(protoInfo.Content, protoInfo.Path);
            collection.Messages.AddRange(model.Messages);
            collection.Enums.AddRange(model.Enums);
            collection.Services.AddRange(model.Services);
        }
            
        return collection;
    }

    /// <summary>
    /// Add unique messages to the combined model
    /// </summary>
    private void AddUniqueMessagesToModel(ProtoModel combinedModel, List<ProtoMessage> allMessages)
    {
        var uniqueMessages = GetUniqueMessages(allMessages);
            
        foreach (var message in uniqueMessages)
        {
            combinedModel.Messages.Add(message);
        }
    }

    /// <summary>
    /// Get unique messages based on full name
    /// </summary>
    private List<ProtoMessage> GetUniqueMessages(List<ProtoMessage> allMessages)
    {
        return allMessages
            .GroupBy(m => m.GetFullName())
            .Select(g => g.First())
            .ToList();
    }

    /// <summary>
    /// Add unique enums to the combined model
    /// </summary>
    private void AddUniqueEnumsToModel(ProtoModel combinedModel, List<ProtoEnum> allEnums)
    {
        var uniqueEnums = GetUniqueEnums(allEnums);
            
        foreach (var enumDef in uniqueEnums)
        {
            combinedModel.Enums.Add(enumDef);
        }
    }

    /// <summary>
    /// Get unique enums based on full name
    /// </summary>
    private List<ProtoEnum> GetUniqueEnums(List<ProtoEnum> allEnums)
    {
        return allEnums
            .GroupBy(e => e.GetFullName())
            .Select(g => g.First())
            .ToList();
    }

    /// <summary>
    /// Process services and add them to the combined model
    /// </summary>
    private void ProcessAndAddServices(ProtoModel combinedModel, List<ProtoService> allServices)
    {
        foreach (var svc in allServices)
        {
            ProcessServiceRpcTypes(combinedModel, svc);
            combinedModel.Services.Add(svc);
        }
    }

    /// <summary>
    /// Process RPC types for a service
    /// </summary>
    private void ProcessServiceRpcTypes(ProtoModel combinedModel, ProtoService svc)
    {
        foreach (var rpc in svc.Rpcs)
        {
            rpc.RequestFullTypename = combinedModel.FindRpcFullTypename(rpc.RequestType);
            rpc.ResponseFullTypename = combinedModel.FindRpcFullTypename(rpc.ResponseType);
        }
    }
}